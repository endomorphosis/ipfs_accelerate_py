"""Independent multi-method repair-candidate portfolio selection (PDR-054).

Interface: ``RepairCandidatePortfolio@1`` / ``RepairCandidateDecision@1``

Compares multiple plausible repair patches by *independent* correctness,
security, and minimality evidence under fixed seeds and budgets. Validation
outcomes are evidence, not weighted authority: hard failures cannot be
averaged away. Candidates remain proposal-only; this module never grants
write, merge, or completion authority.

Lanes (when supported / required):

* property-based, fuzz, concolic (optional capability; recorded when unavailable)
* mutation testing against an independent oracle
* differential and metamorphic checks
* sanitizers
* static / model checks
* proof and security gates

Hard admission rejects self-authored tests and candidate-as-oracle. Only
hard-admissible candidates are ranked by minimal blast radius, then resource
cost. When none remain, the portfolio abstains. Selection and replay identities
are content-addressed and identity-stable under identical frozen inputs.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Final, Protocol

from ..proof.formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    content_identity,
)

# ---------------------------------------------------------------------------
# Schema / interface constants
# ---------------------------------------------------------------------------

REPAIR_CANDIDATE_PORTFOLIO_INTERFACE: Final[str] = "RepairCandidatePortfolio@1"
REPAIR_CANDIDATE_DECISION_INTERFACE: Final[str] = "RepairCandidateDecision@1"
REPAIR_CANDIDATE_PORTFOLIO_VERSION: Final[str] = "1.0.0"
CONTRACT_VERSION: Final[int] = 1

PORTFOLIO_REQUEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/repair-candidate-portfolio-request@1"
)
PORTFOLIO_CANDIDATE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/repair-portfolio-candidate@1"
)
PORTFOLIO_LANE_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/repair-portfolio-lane-result@1"
)
PORTFOLIO_EVALUATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/repair-portfolio-candidate-evaluation@1"
)
PORTFOLIO_DECISION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/repair-candidate-decision@1"
)
PORTFOLIO_BUDGET_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/repair-portfolio-budget@1"
)
INDEPENDENT_ORACLE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/independent-validation-oracle@1"
)
HARD_OBLIGATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/repair-portfolio-hard-obligation@1"
)

PRODUCER_ID: Final[str] = "repair-candidate-portfolio@1"

MAX_CANDIDATES: Final[int] = 64
MAX_LANES: Final[int] = 32
MAX_OBLIGATIONS: Final[int] = 128
MAX_TEXT_BYTES: Final[int] = 4096
MAX_PATH_COUNT: Final[int] = 4096
MAX_ID_BYTES: Final[int] = 512
MAX_REASON_CODES: Final[int] = 64
DEFAULT_SEED: Final[int] = 0x504452_054  # "PDR" + 054
DEFAULT_MAX_PROPERTY_CASES: Final[int] = 256
DEFAULT_MAX_FUZZ_INPUTS: Final[int] = 1024
DEFAULT_MAX_CONCOLIC_PATHS: Final[int] = 128
DEFAULT_MAX_MUTATION_OPS: Final[int] = 64
DEFAULT_MAX_WALL_MS: Final[int] = 60_000
DEFAULT_MAX_RESOURCE_COST: Final[int] = 1_000_000

# Oracle sources that are never independent of the candidate under test.
_FORBIDDEN_ORACLE_SOURCES: Final[frozenset[str]] = frozenset(
    {
        "candidate",
        "candidate_authored",
        "candidate_generated",
        "self",
        "self_authored",
        "patch",
        "proposal",
        "model",
        "llm",
        "synthesized_by_candidate",
    }
)

# Lanes that may be optional when capability is absent (recorded, not silent pass).
OPTIONAL_CAPABILITY_LANES: Final[frozenset[str]] = frozenset(
    {
        "property_based",
        "fuzz",
        "concolic",
    }
)


# ---------------------------------------------------------------------------
# Errors / vocabularies
# ---------------------------------------------------------------------------


class RepairCandidatePortfolioError(ContractValidationError):
    """Portfolio request or evaluation is malformed or unsafe."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: "PortfolioReason | str" = "",
    ) -> None:
        super().__init__(message)
        code = (
            reason_code.value
            if isinstance(reason_code, PortfolioReason)
            else str(reason_code or "")
        )
        self.reason_code = code


class PortfolioAuthorityError(RepairCandidatePortfolioError):
    """Raised when a candidate or decision claims forbidden authority."""


class PortfolioBoundsError(RepairCandidatePortfolioError):
    """Raised when a budget or collection bound is exceeded."""


class PortfolioLane(str, Enum):
    """Closed set of independent multi-method validation lanes."""

    PROPERTY_BASED = "property_based"
    FUZZ = "fuzz"
    CONCOLIC = "concolic"
    MUTATION = "mutation"
    DIFFERENTIAL = "differential"
    METAMORPHIC = "metamorphic"
    SANITIZER = "sanitizer"
    STATIC = "static"
    MODEL = "model"
    PROOF = "proof"
    SECURITY = "security"


# Canonical evaluation order (stable for replay identity).
PORTFOLIO_LANE_ORDER: Final[tuple[PortfolioLane, ...]] = (
    PortfolioLane.PROPERTY_BASED,
    PortfolioLane.FUZZ,
    PortfolioLane.CONCOLIC,
    PortfolioLane.MUTATION,
    PortfolioLane.DIFFERENTIAL,
    PortfolioLane.METAMORPHIC,
    PortfolioLane.SANITIZER,
    PortfolioLane.STATIC,
    PortfolioLane.MODEL,
    PortfolioLane.PROOF,
    PortfolioLane.SECURITY,
)

# Default hard lanes: fail closed unless an obligation set overrides.
DEFAULT_HARD_LANES: Final[frozenset[PortfolioLane]] = frozenset(
    {
        PortfolioLane.MUTATION,
        PortfolioLane.DIFFERENTIAL,
        PortfolioLane.METAMORPHIC,
        PortfolioLane.STATIC,
        PortfolioLane.MODEL,
        PortfolioLane.PROOF,
        PortfolioLane.SECURITY,
    }
)


class LaneOutcome(str, Enum):
    """Closed outcomes for one validation lane observation."""

    PASS = "pass"
    FAIL = "fail"
    FLAKY = "flaky"
    UNAVAILABLE = "unavailable"
    SKIPPED = "skipped"


class PortfolioDisposition(str, Enum):
    """Closed portfolio decision outcomes."""

    SELECTED = "selected"
    ABSTAIN = "abstain"
    REJECT = "reject"


class PortfolioReason(str, Enum):
    """Stable fail-closed reason codes for PDR-054."""

    OK = "ok"
    NO_HARD_ADMISSIBLE = "no_hard_admissible_candidate"
    HARD_OBLIGATION_FAILED = "hard_obligation_failed"
    HARD_OBLIGATION_UNAVAILABLE = "hard_obligation_unavailable"
    HARD_OBLIGATION_FLAKY = "hard_obligation_flaky"
    SELF_AUTHORED_TEST = "self_authored_test"
    CANDIDATE_AS_ORACLE = "candidate_as_oracle"
    ORACLE_NOT_INDEPENDENT = "oracle_not_independent"
    MISSING_ORACLE = "missing_independent_oracle"
    WEIGHTED_AUTHORITY = "weighted_authority_rejected"
    AUTHORITY_CLAIM = "authority_claim"
    PROPOSAL_ONLY = "proposal_only"
    BUDGET_EXCEEDED = "budget_exceeded"
    BOUNDS_EXCEEDED = "bounds_exceeded"
    MALFORMED_INPUT = "malformed_input"
    DUPLICATE_CANDIDATE = "duplicate_candidate"
    EMPTY_PORTFOLIO = "empty_portfolio"
    LANE_FAIL = "lane_fail"
    SELECTION_REPLAY_MISMATCH = "selection_replay_mismatch"
    NO_CANDIDATES = "no_candidates"
    CORRECT_ABSTENTION = "correct_abstention"
    HARD_FAILURE_NOT_AVERAGED = "hard_failure_not_averaged"
    SOFT_DEBT_ONLY = "soft_debt_only"
    ALL_HARD_OBLIGATIONS_MET = "all_hard_obligations_met"
    MINIMAL_BLAST_RADIUS = "minimal_blast_radius"
    MINIMAL_RESOURCE_COST = "minimal_resource_cost"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text(
    value: Any,
    name: str,
    *,
    required: bool = True,
    limit: int = MAX_TEXT_BYTES,
) -> str:
    if value is None:
        text = ""
    elif isinstance(value, str):
        text = value.strip()
    else:
        text = str(value).strip()
    if required and not text:
        raise RepairCandidatePortfolioError(
            f"{name} is required",
            reason_code=PortfolioReason.MALFORMED_INPUT,
        )
    if "\x00" in text or len(text.encode("utf-8")) > limit:
        raise PortfolioBoundsError(
            f"{name} is invalid or exceeds its bound",
            reason_code=PortfolioReason.BOUNDS_EXCEEDED,
        )
    return text


def _optional_text(value: Any, name: str, *, limit: int = MAX_TEXT_BYTES) -> str:
    if value is None or value == "":
        return ""
    return _text(value, name, required=True, limit=limit)


def _nonneg_int(
    value: Any,
    name: str,
    *,
    maximum: int | None = None,
    default: int | None = None,
) -> int:
    if value is None and default is not None:
        value = default
    if isinstance(value, bool) or not isinstance(value, int):
        raise RepairCandidatePortfolioError(
            f"{name} must be a non-negative integer",
            reason_code=PortfolioReason.MALFORMED_INPUT,
        )
    if value < 0:
        raise RepairCandidatePortfolioError(
            f"{name} must be a non-negative integer",
            reason_code=PortfolioReason.MALFORMED_INPUT,
        )
    if maximum is not None and value > maximum:
        raise PortfolioBoundsError(
            f"{name} exceeds bound {maximum}",
            reason_code=PortfolioReason.BOUNDS_EXCEEDED,
        )
    return value


def _ids(
    values: Any,
    name: str,
    *,
    required: bool = False,
    limit: int = MAX_OBLIGATIONS,
    preserve_order: bool = False,
) -> tuple[str, ...]:
    if values is None:
        items: Sequence[Any] = ()
    elif isinstance(values, str):
        items = (values,)
    elif isinstance(values, Sequence) and not isinstance(values, (bytes, bytearray)):
        items = values
    else:
        raise RepairCandidatePortfolioError(
            f"{name} must be a sequence of identifiers",
            reason_code=PortfolioReason.MALFORMED_INPUT,
        )
    out: list[str] = []
    seen: set[str] = set()
    for raw in items:
        text = _text(raw, name, limit=MAX_ID_BYTES)
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
        if len(out) > limit:
            raise PortfolioBoundsError(
                f"{name} exceeds bound {limit}",
                reason_code=PortfolioReason.BOUNDS_EXCEEDED,
            )
    if required and not out:
        raise RepairCandidatePortfolioError(
            f"{name} must not be empty",
            reason_code=PortfolioReason.MALFORMED_INPUT,
        )
    if preserve_order:
        return tuple(out)
    return tuple(sorted(out))


def _paths(values: Any, name: str = "changed_paths") -> tuple[str, ...]:
    ids = _ids(values, name, limit=MAX_PATH_COUNT, preserve_order=True)
    normalized: list[str] = []
    seen: set[str] = set()
    for path in ids:
        cleaned = path.replace("\\", "/").lstrip("./")
        if (
            not cleaned
            or cleaned.startswith("/")
            or ".." in cleaned.split("/")
            or "\x00" in cleaned
        ):
            raise RepairCandidatePortfolioError(
                f"{name} contains an unsafe path",
                reason_code=PortfolioReason.MALFORMED_INPUT,
            )
        if cleaned not in seen:
            seen.add(cleaned)
            normalized.append(cleaned)
    return tuple(normalized)


def _enum(value: Any, enum_cls: type[Enum], name: str) -> Any:
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        try:
            return enum_cls(value.strip())
        except ValueError as exc:
            raise RepairCandidatePortfolioError(
                f"{name} is not a valid {enum_cls.__name__}",
                reason_code=PortfolioReason.MALFORMED_INPUT,
            ) from exc
    raise RepairCandidatePortfolioError(
        f"{name} is not a valid {enum_cls.__name__}",
        reason_code=PortfolioReason.MALFORMED_INPUT,
    )


def _bool(value: Any, name: str, *, default: bool | None = None) -> bool:
    if value is None and default is not None:
        return default
    if not isinstance(value, bool):
        raise RepairCandidatePortfolioError(
            f"{name} must be a boolean",
            reason_code=PortfolioReason.MALFORMED_INPUT,
        )
    return value


def _mapping_proxy(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise RepairCandidatePortfolioError(
            f"{name} must be a mapping",
            reason_code=PortfolioReason.MALFORMED_INPUT,
        )
    return MappingProxyType(dict(value))


def _lane(value: Any) -> PortfolioLane:
    return _enum(value, PortfolioLane, "lane")


def _normalize_oracle_source(source: str) -> str:
    return source.strip().casefold().replace("-", "_").replace(" ", "_")


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PortfolioSeedBudget(CanonicalContract):
    """Fixed seeds and budgets bound into every portfolio evaluation."""

    SCHEMA: ClassVar[str] = PORTFOLIO_BUDGET_SCHEMA

    seed: int = DEFAULT_SEED
    max_property_cases: int = DEFAULT_MAX_PROPERTY_CASES
    max_fuzz_inputs: int = DEFAULT_MAX_FUZZ_INPUTS
    max_concolic_paths: int = DEFAULT_MAX_CONCOLIC_PATHS
    max_mutation_ops: int = DEFAULT_MAX_MUTATION_OPS
    max_wall_ms: int = DEFAULT_MAX_WALL_MS
    max_resource_cost: int = DEFAULT_MAX_RESOURCE_COST

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "seed", _nonneg_int(self.seed, "seed", maximum=2**63 - 1)
        )
        for name, maximum in (
            ("max_property_cases", 1_000_000),
            ("max_fuzz_inputs", 10_000_000),
            ("max_concolic_paths", 1_000_000),
            ("max_mutation_ops", 1_000_000),
            ("max_wall_ms", 3_600_000),
            ("max_resource_cost", 100_000_000),
        ):
            object.__setattr__(
                self,
                name,
                _nonneg_int(getattr(self, name), name, maximum=maximum),
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "seed": self.seed,
            "max_property_cases": self.max_property_cases,
            "max_fuzz_inputs": self.max_fuzz_inputs,
            "max_concolic_paths": self.max_concolic_paths,
            "max_mutation_ops": self.max_mutation_ops,
            "max_wall_ms": self.max_wall_ms,
            "max_resource_cost": self.max_resource_cost,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "PortfolioSeedBudget":
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise RepairCandidatePortfolioError(
                "budget must be a mapping",
                reason_code=PortfolioReason.MALFORMED_INPUT,
            )
        if payload.get("schema") not in {None, "", PORTFOLIO_BUDGET_SCHEMA}:
            raise RepairCandidatePortfolioError(
                "unsupported portfolio budget schema",
                reason_code=PortfolioReason.MALFORMED_INPUT,
            )
        return cls(
            seed=int(payload.get("seed", DEFAULT_SEED)),
            max_property_cases=int(
                payload.get("max_property_cases", DEFAULT_MAX_PROPERTY_CASES)
            ),
            max_fuzz_inputs=int(
                payload.get("max_fuzz_inputs", DEFAULT_MAX_FUZZ_INPUTS)
            ),
            max_concolic_paths=int(
                payload.get("max_concolic_paths", DEFAULT_MAX_CONCOLIC_PATHS)
            ),
            max_mutation_ops=int(
                payload.get("max_mutation_ops", DEFAULT_MAX_MUTATION_OPS)
            ),
            max_wall_ms=int(payload.get("max_wall_ms", DEFAULT_MAX_WALL_MS)),
            max_resource_cost=int(
                payload.get("max_resource_cost", DEFAULT_MAX_RESOURCE_COST)
            ),
        )


@dataclass(frozen=True)
class IndependentOracle(CanonicalContract):
    """Independent acceptance oracle — never candidate-authored or self-validating."""

    SCHEMA: ClassVar[str] = INDEPENDENT_ORACLE_SCHEMA

    oracle_id: str
    source: str
    producer_id: str
    expectation_ids: tuple[str, ...] = ()
    test_ids: tuple[str, ...] = ()
    root_bindings: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "oracle_id", _text(self.oracle_id, "oracle_id", limit=MAX_ID_BYTES)
        )
        source = _text(self.source, "source", limit=MAX_ID_BYTES)
        object.__setattr__(self, "source", source)
        object.__setattr__(
            self,
            "producer_id",
            _text(self.producer_id, "producer_id", limit=MAX_ID_BYTES),
        )
        object.__setattr__(
            self,
            "expectation_ids",
            _ids(self.expectation_ids, "expectation_ids", preserve_order=True),
        )
        object.__setattr__(
            self, "test_ids", _ids(self.test_ids, "test_ids", preserve_order=True)
        )
        bindings = tuple(self.root_bindings or ())
        normalized: list[tuple[str, str]] = []
        for item in bindings:
            if (
                not isinstance(item, Sequence)
                or isinstance(item, (str, bytes))
                or len(item) != 2
            ):
                raise RepairCandidatePortfolioError(
                    "root_bindings entries must be (key, value) pairs",
                    reason_code=PortfolioReason.MALFORMED_INPUT,
                )
            normalized.append(
                (
                    _text(item[0], "root_bindings.key", limit=MAX_ID_BYTES),
                    _text(item[1], "root_bindings.value", limit=MAX_ID_BYTES),
                )
            )
        object.__setattr__(
            self,
            "root_bindings",
            tuple(sorted(normalized, key=lambda pair: pair[0])),
        )
        if not self.is_independent():
            raise RepairCandidatePortfolioError(
                "oracle source is not independent of the candidate",
                reason_code=PortfolioReason.ORACLE_NOT_INDEPENDENT,
            )

    def is_independent(self) -> bool:
        source = _normalize_oracle_source(self.source)
        producer = _normalize_oracle_source(self.producer_id)
        if source in _FORBIDDEN_ORACLE_SOURCES:
            return False
        if producer in _FORBIDDEN_ORACLE_SOURCES:
            return False
        if "candidate" in source or "self_authored" in source:
            return False
        if "candidate" in producer or "self_authored" in producer:
            return False
        return True

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "oracle_id": self.oracle_id,
            "source": self.source,
            "producer_id": self.producer_id,
            "expectation_ids": list(self.expectation_ids),
            "test_ids": list(self.test_ids),
            "root_bindings": [[k, v] for k, v in self.root_bindings],
            "independent": True,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "IndependentOracle":
        if not isinstance(payload, Mapping):
            raise RepairCandidatePortfolioError(
                "oracle must be a mapping",
                reason_code=PortfolioReason.MALFORMED_INPUT,
            )
        if payload.get("schema") not in {None, "", INDEPENDENT_ORACLE_SCHEMA}:
            raise RepairCandidatePortfolioError(
                "unsupported independent oracle schema",
                reason_code=PortfolioReason.MALFORMED_INPUT,
            )
        bindings_raw = payload.get("root_bindings") or ()
        bindings: list[tuple[str, str]] = []
        for item in bindings_raw:
            if isinstance(item, Mapping):
                bindings.append(
                    (str(item.get("key") or ""), str(item.get("value") or ""))
                )
            else:
                bindings.append((str(item[0]), str(item[1])))
        return cls(
            oracle_id=str(payload.get("oracle_id") or ""),
            source=str(payload.get("source") or ""),
            producer_id=str(payload.get("producer_id") or ""),
            expectation_ids=tuple(payload.get("expectation_ids") or ()),
            test_ids=tuple(payload.get("test_ids") or ()),
            root_bindings=tuple(bindings),
        )


@dataclass(frozen=True)
class HardObligation(CanonicalContract):
    """A required validation obligation that cannot be averaged away."""

    SCHEMA: ClassVar[str] = HARD_OBLIGATION_SCHEMA

    obligation_id: str
    lane: PortfolioLane
    required: bool = True
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "obligation_id",
            _text(self.obligation_id, "obligation_id", limit=MAX_ID_BYTES),
        )
        object.__setattr__(self, "lane", _lane(self.lane))
        object.__setattr__(self, "required", _bool(self.required, "required", default=True))
        object.__setattr__(
            self,
            "description",
            _optional_text(self.description, "description"),
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "obligation_id": self.obligation_id,
            "lane": self.lane.value,
            "required": self.required,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HardObligation":
        if not isinstance(payload, Mapping):
            raise RepairCandidatePortfolioError(
                "hard obligation must be a mapping",
                reason_code=PortfolioReason.MALFORMED_INPUT,
            )
        return cls(
            obligation_id=str(payload.get("obligation_id") or ""),
            lane=str(payload.get("lane") or ""),
            required=bool(payload.get("required", True)),
            description=str(payload.get("description") or ""),
        )


@dataclass(frozen=True)
class LaneObservation:
    """Structured observation for one validation lane under a fixed seed/budget.

    Callers (worktree adapters, hermetic harnesses, tests) supply these
    observations. The portfolio never treats them as weighted scores.
    """

    supported: bool = True
    status: str = "pass"
    cases_run: int = 0
    budget_used: int = 0
    uses_self_authored_tests: bool = False
    oracle_id: str = ""
    candidate_claims_oracle: bool = False
    evidence_refs: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    soft_score: int | None = None  # ignored for hard admission / ranking authority

    def __post_init__(self) -> None:
        object.__setattr__(self, "supported", _bool(self.supported, "supported", default=True))
        object.__setattr__(
            self,
            "status",
            _text(self.status, "status", limit=64).casefold(),
        )
        object.__setattr__(
            self, "cases_run", _nonneg_int(self.cases_run, "cases_run", maximum=10_000_000)
        )
        object.__setattr__(
            self,
            "budget_used",
            _nonneg_int(self.budget_used, "budget_used", maximum=10_000_000),
        )
        object.__setattr__(
            self,
            "uses_self_authored_tests",
            _bool(
                self.uses_self_authored_tests,
                "uses_self_authored_tests",
                default=False,
            ),
        )
        object.__setattr__(
            self, "oracle_id", _optional_text(self.oracle_id, "oracle_id", limit=MAX_ID_BYTES)
        )
        object.__setattr__(
            self,
            "candidate_claims_oracle",
            _bool(
                self.candidate_claims_oracle,
                "candidate_claims_oracle",
                default=False,
            ),
        )
        object.__setattr__(
            self,
            "evidence_refs",
            _ids(self.evidence_refs, "evidence_refs", preserve_order=True),
        )
        object.__setattr__(
            self,
            "reason_codes",
            _ids(self.reason_codes, "reason_codes", preserve_order=True, limit=MAX_REASON_CODES),
        )
        if self.soft_score is not None:
            object.__setattr__(
                self,
                "soft_score",
                _nonneg_int(self.soft_score, "soft_score", maximum=1_000_000),
            )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "LaneObservation":
        if payload is None:
            return cls(supported=False, status="unavailable")
        if not isinstance(payload, Mapping):
            raise RepairCandidatePortfolioError(
                "lane observation must be a mapping",
                reason_code=PortfolioReason.MALFORMED_INPUT,
            )
        return cls(
            supported=bool(payload.get("supported", True)),
            status=str(payload.get("status") or "pass"),
            cases_run=int(payload.get("cases_run") or 0),
            budget_used=int(payload.get("budget_used") or 0),
            uses_self_authored_tests=bool(
                payload.get("uses_self_authored_tests") or False
            ),
            oracle_id=str(payload.get("oracle_id") or ""),
            candidate_claims_oracle=bool(
                payload.get("candidate_claims_oracle") or False
            ),
            evidence_refs=tuple(payload.get("evidence_refs") or ()),
            reason_codes=tuple(payload.get("reason_codes") or ()),
            soft_score=(
                int(payload["soft_score"])
                if payload.get("soft_score") is not None
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "supported": self.supported,
            "status": self.status,
            "cases_run": self.cases_run,
            "budget_used": self.budget_used,
            "uses_self_authored_tests": self.uses_self_authored_tests,
            "oracle_id": self.oracle_id,
            "candidate_claims_oracle": self.candidate_claims_oracle,
            "evidence_refs": list(self.evidence_refs),
            "reason_codes": list(self.reason_codes),
            "soft_score": self.soft_score,
        }


@dataclass(frozen=True)
class PortfolioCandidate(CanonicalContract):
    """One proposal-only repair candidate under portfolio evaluation."""

    SCHEMA: ClassVar[str] = PORTFOLIO_CANDIDATE_SCHEMA

    candidate_id: str
    patch_cid: str
    overlay_cid: str = ""
    changed_paths: tuple[str, ...] = ()
    blast_radius: int = 0
    resource_cost: int = 0
    obligation_refs: tuple[str, ...] = ()
    authored_test_ids: tuple[str, ...] = ()
    claimed_oracle_ids: tuple[str, ...] = ()
    lane_support: Mapping[str, bool] = field(default_factory=dict)
    lane_observations: Mapping[str, Any] = field(default_factory=dict)
    proposal_only: bool = True
    write_authority: bool = False
    semantic_authority: bool = False
    grants_proof_authority: bool = False
    producer_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_id",
            _text(self.candidate_id, "candidate_id", limit=MAX_ID_BYTES),
        )
        object.__setattr__(
            self, "patch_cid", _text(self.patch_cid, "patch_cid", limit=MAX_ID_BYTES)
        )
        object.__setattr__(
            self,
            "overlay_cid",
            _optional_text(self.overlay_cid, "overlay_cid", limit=MAX_ID_BYTES),
        )
        object.__setattr__(self, "changed_paths", _paths(self.changed_paths))
        object.__setattr__(
            self,
            "blast_radius",
            _nonneg_int(self.blast_radius, "blast_radius", maximum=10_000_000),
        )
        object.__setattr__(
            self,
            "resource_cost",
            _nonneg_int(self.resource_cost, "resource_cost", maximum=100_000_000),
        )
        object.__setattr__(
            self,
            "obligation_refs",
            _ids(self.obligation_refs, "obligation_refs", preserve_order=True),
        )
        object.__setattr__(
            self,
            "authored_test_ids",
            _ids(self.authored_test_ids, "authored_test_ids", preserve_order=True),
        )
        object.__setattr__(
            self,
            "claimed_oracle_ids",
            _ids(self.claimed_oracle_ids, "claimed_oracle_ids", preserve_order=True),
        )
        support_raw = self.lane_support if self.lane_support is not None else {}
        if not isinstance(support_raw, Mapping):
            raise RepairCandidatePortfolioError(
                "lane_support must be a mapping",
                reason_code=PortfolioReason.MALFORMED_INPUT,
            )
        support: dict[str, bool] = {}
        for key, value in support_raw.items():
            lane_key = str(key).strip()
            if not lane_key:
                continue
            # Accept PortfolioLane values or strings.
            try:
                lane_key = _lane(lane_key).value
            except RepairCandidatePortfolioError:
                pass
            if not isinstance(value, bool):
                raise RepairCandidatePortfolioError(
                    "lane_support values must be booleans",
                    reason_code=PortfolioReason.MALFORMED_INPUT,
                )
            support[lane_key] = value
        object.__setattr__(self, "lane_support", MappingProxyType(support))

        obs_raw = self.lane_observations if self.lane_observations is not None else {}
        if not isinstance(obs_raw, Mapping):
            raise RepairCandidatePortfolioError(
                "lane_observations must be a mapping",
                reason_code=PortfolioReason.MALFORMED_INPUT,
            )
        observations: dict[str, dict[str, Any]] = {}
        for key, value in obs_raw.items():
            lane_key = str(key).strip()
            try:
                lane_key = _lane(lane_key).value
            except RepairCandidatePortfolioError:
                pass
            if isinstance(value, LaneObservation):
                observations[lane_key] = value.to_dict()
            elif isinstance(value, Mapping):
                observations[lane_key] = dict(value)
            else:
                raise RepairCandidatePortfolioError(
                    "lane_observations values must be mappings",
                    reason_code=PortfolioReason.MALFORMED_INPUT,
                )
        object.__setattr__(self, "lane_observations", MappingProxyType(observations))

        if self.proposal_only is not True:
            raise PortfolioAuthorityError(
                "candidates must remain proposal-only",
                reason_code=PortfolioReason.PROPOSAL_ONLY,
            )
        object.__setattr__(self, "proposal_only", True)
        for name in ("write_authority", "semantic_authority", "grants_proof_authority"):
            if getattr(self, name) is not False:
                raise PortfolioAuthorityError(
                    f"candidate cannot claim {name}",
                    reason_code=PortfolioReason.AUTHORITY_CLAIM,
                )
            object.__setattr__(self, name, False)
        object.__setattr__(
            self,
            "producer_id",
            _optional_text(self.producer_id, "producer_id", limit=MAX_ID_BYTES),
        )

    def observation_for(self, lane: PortfolioLane) -> LaneObservation:
        raw = self.lane_observations.get(lane.value)
        if raw is None:
            supported = self.lane_support.get(lane.value, True)
            if not supported:
                return LaneObservation(supported=False, status="unavailable")
            return LaneObservation(supported=True, status="pass")
        obs = LaneObservation.from_mapping(raw)
        # lane_support overrides when explicitly false.
        if self.lane_support.get(lane.value) is False:
            return LaneObservation(
                supported=False,
                status="unavailable",
                cases_run=obs.cases_run,
                budget_used=obs.budget_used,
                uses_self_authored_tests=obs.uses_self_authored_tests,
                oracle_id=obs.oracle_id,
                candidate_claims_oracle=obs.candidate_claims_oracle,
                evidence_refs=obs.evidence_refs,
                reason_codes=obs.reason_codes,
                soft_score=obs.soft_score,
            )
        return obs

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "candidate_id": self.candidate_id,
            "patch_cid": self.patch_cid,
            "overlay_cid": self.overlay_cid,
            "changed_paths": list(self.changed_paths),
            "blast_radius": self.blast_radius,
            "resource_cost": self.resource_cost,
            "obligation_refs": list(self.obligation_refs),
            "authored_test_ids": list(self.authored_test_ids),
            "claimed_oracle_ids": list(self.claimed_oracle_ids),
            "lane_support": dict(self.lane_support),
            "lane_observations": {
                key: dict(value) for key, value in self.lane_observations.items()
            },
            "proposal_only": True,
            "write_authority": False,
            "semantic_authority": False,
            "grants_proof_authority": False,
            "producer_id": self.producer_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PortfolioCandidate":
        if not isinstance(payload, Mapping):
            raise RepairCandidatePortfolioError(
                "candidate must be a mapping",
                reason_code=PortfolioReason.MALFORMED_INPUT,
            )
        if payload.get("schema") not in {None, "", PORTFOLIO_CANDIDATE_SCHEMA}:
            raise RepairCandidatePortfolioError(
                "unsupported portfolio candidate schema",
                reason_code=PortfolioReason.MALFORMED_INPUT,
            )
        return cls(
            candidate_id=str(payload.get("candidate_id") or ""),
            patch_cid=str(payload.get("patch_cid") or ""),
            overlay_cid=str(payload.get("overlay_cid") or ""),
            changed_paths=tuple(payload.get("changed_paths") or ()),
            blast_radius=int(payload.get("blast_radius") or 0),
            resource_cost=int(payload.get("resource_cost") or 0),
            obligation_refs=tuple(payload.get("obligation_refs") or ()),
            authored_test_ids=tuple(payload.get("authored_test_ids") or ()),
            claimed_oracle_ids=tuple(payload.get("claimed_oracle_ids") or ()),
            lane_support=dict(payload.get("lane_support") or {}),
            lane_observations=dict(payload.get("lane_observations") or {}),
            producer_id=str(payload.get("producer_id") or ""),
        )


@dataclass(frozen=True)
class LaneResult(CanonicalContract):
    """Sealed result for one validation lane on one candidate."""

    SCHEMA: ClassVar[str] = PORTFOLIO_LANE_RESULT_SCHEMA

    lane: PortfolioLane
    outcome: LaneOutcome
    hard: bool
    seed: int
    budget_used: int
    cases_run: int = 0
    evidence_id: str = ""
    reason_codes: tuple[str, ...] = ()
    oracle_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "lane", _lane(self.lane))
        object.__setattr__(
            self, "outcome", _enum(self.outcome, LaneOutcome, "outcome")
        )
        object.__setattr__(self, "hard", _bool(self.hard, "hard"))
        object.__setattr__(self, "seed", _nonneg_int(self.seed, "seed", maximum=2**63 - 1))
        object.__setattr__(
            self,
            "budget_used",
            _nonneg_int(self.budget_used, "budget_used", maximum=10_000_000),
        )
        object.__setattr__(
            self, "cases_run", _nonneg_int(self.cases_run, "cases_run", maximum=10_000_000)
        )
        object.__setattr__(
            self,
            "evidence_id",
            _optional_text(self.evidence_id, "evidence_id", limit=MAX_ID_BYTES),
        )
        object.__setattr__(
            self,
            "reason_codes",
            _ids(self.reason_codes, "reason_codes", preserve_order=True, limit=MAX_REASON_CODES),
        )
        object.__setattr__(
            self, "oracle_id", _optional_text(self.oracle_id, "oracle_id", limit=MAX_ID_BYTES)
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "lane": self.lane.value,
            "outcome": self.outcome.value,
            "hard": self.hard,
            "seed": self.seed,
            "budget_used": self.budget_used,
            "cases_run": self.cases_run,
            "evidence_id": self.evidence_id,
            "reason_codes": list(self.reason_codes),
            "oracle_id": self.oracle_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LaneResult":
        return cls(
            lane=str(payload.get("lane") or ""),
            outcome=str(payload.get("outcome") or ""),
            hard=bool(payload.get("hard")),
            seed=int(payload.get("seed") or 0),
            budget_used=int(payload.get("budget_used") or 0),
            cases_run=int(payload.get("cases_run") or 0),
            evidence_id=str(payload.get("evidence_id") or ""),
            reason_codes=tuple(payload.get("reason_codes") or ()),
            oracle_id=str(payload.get("oracle_id") or ""),
        )


@dataclass(frozen=True)
class CandidateEvaluation(CanonicalContract):
    """Aggregated multi-method evaluation for one repair candidate."""

    SCHEMA: ClassVar[str] = PORTFOLIO_EVALUATION_SCHEMA

    candidate_id: str
    hard_admissible: bool
    blast_radius: int
    resource_cost: int
    lane_results: tuple[LaneResult, ...] = ()
    hard_failures: tuple[str, ...] = ()
    flaky_lanes: tuple[str, ...] = ()
    unavailable_lanes: tuple[str, ...] = ()
    rejection_reasons: tuple[str, ...] = ()
    soft_debt: tuple[str, ...] = ()
    ranking_key: tuple[int, int, str] = (0, 0, "")

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_id",
            _text(self.candidate_id, "candidate_id", limit=MAX_ID_BYTES),
        )
        object.__setattr__(
            self, "hard_admissible", _bool(self.hard_admissible, "hard_admissible")
        )
        object.__setattr__(
            self,
            "blast_radius",
            _nonneg_int(self.blast_radius, "blast_radius", maximum=10_000_000),
        )
        object.__setattr__(
            self,
            "resource_cost",
            _nonneg_int(self.resource_cost, "resource_cost", maximum=100_000_000),
        )
        results = tuple(self.lane_results or ())
        for item in results:
            if not isinstance(item, LaneResult):
                raise RepairCandidatePortfolioError(
                    "lane_results must contain LaneResult values",
                    reason_code=PortfolioReason.MALFORMED_INPUT,
                )
        object.__setattr__(self, "lane_results", results)
        object.__setattr__(
            self,
            "hard_failures",
            _ids(self.hard_failures, "hard_failures", preserve_order=True),
        )
        object.__setattr__(
            self,
            "flaky_lanes",
            _ids(self.flaky_lanes, "flaky_lanes", preserve_order=True),
        )
        object.__setattr__(
            self,
            "unavailable_lanes",
            _ids(self.unavailable_lanes, "unavailable_lanes", preserve_order=True),
        )
        object.__setattr__(
            self,
            "rejection_reasons",
            _ids(self.rejection_reasons, "rejection_reasons", preserve_order=True),
        )
        object.__setattr__(
            self, "soft_debt", _ids(self.soft_debt, "soft_debt", preserve_order=True)
        )
        key = self.ranking_key
        if not isinstance(key, tuple) or len(key) != 3:
            key = (self.blast_radius, self.resource_cost, self.candidate_id)
        object.__setattr__(
            self,
            "ranking_key",
            (int(key[0]), int(key[1]), str(key[2])),
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "candidate_id": self.candidate_id,
            "hard_admissible": self.hard_admissible,
            "blast_radius": self.blast_radius,
            "resource_cost": self.resource_cost,
            "lane_results": [item.to_dict() for item in self.lane_results],
            "hard_failures": list(self.hard_failures),
            "flaky_lanes": list(self.flaky_lanes),
            "unavailable_lanes": list(self.unavailable_lanes),
            "rejection_reasons": list(self.rejection_reasons),
            "soft_debt": list(self.soft_debt),
            "ranking_key": [
                self.ranking_key[0],
                self.ranking_key[1],
                self.ranking_key[2],
            ],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CandidateEvaluation":
        results = tuple(
            LaneResult.from_dict(item)
            for item in (payload.get("lane_results") or ())
        )
        key_raw = payload.get("ranking_key") or (
            payload.get("blast_radius") or 0,
            payload.get("resource_cost") or 0,
            payload.get("candidate_id") or "",
        )
        return cls(
            candidate_id=str(payload.get("candidate_id") or ""),
            hard_admissible=bool(payload.get("hard_admissible")),
            blast_radius=int(payload.get("blast_radius") or 0),
            resource_cost=int(payload.get("resource_cost") or 0),
            lane_results=results,
            hard_failures=tuple(payload.get("hard_failures") or ()),
            flaky_lanes=tuple(payload.get("flaky_lanes") or ()),
            unavailable_lanes=tuple(payload.get("unavailable_lanes") or ()),
            rejection_reasons=tuple(payload.get("rejection_reasons") or ()),
            soft_debt=tuple(payload.get("soft_debt") or ()),
            ranking_key=(int(key_raw[0]), int(key_raw[1]), str(key_raw[2])),
        )


@dataclass(frozen=True)
class RepairCandidateDecision(CanonicalContract):
    """Sealed multi-method portfolio decision with selection/replay identity."""

    SCHEMA: ClassVar[str] = PORTFOLIO_DECISION_SCHEMA
    INTERFACE: ClassVar[str] = REPAIR_CANDIDATE_DECISION_INTERFACE

    disposition: PortfolioDisposition
    reason_codes: tuple[str, ...]
    evaluations: tuple[CandidateEvaluation, ...]
    ranked_admissible: tuple[str, ...] = ()
    selected_candidate_id: str = ""
    selection_identity: str = ""
    replay_identity: str = ""
    flaky_lanes: tuple[str, ...] = ()
    unavailable_lanes: tuple[str, ...] = ()
    hard_obligation_ids: tuple[str, ...] = ()
    seed: int = DEFAULT_SEED
    budget: PortfolioSeedBudget = field(default_factory=PortfolioSeedBudget)
    oracle_id: str = ""
    proposal_only: bool = True
    write_authority: bool = False
    semantic_authority: bool = False
    grants_completion_authority: bool = False
    weighted_authority_used: bool = False
    producer_id: str = PRODUCER_ID
    interface: str = REPAIR_CANDIDATE_DECISION_INTERFACE

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, PortfolioDisposition, "disposition"),
        )
        object.__setattr__(
            self,
            "reason_codes",
            _ids(
                self.reason_codes,
                "reason_codes",
                required=True,
                preserve_order=True,
                limit=MAX_REASON_CODES,
            ),
        )
        evaluations = tuple(self.evaluations or ())
        for item in evaluations:
            if not isinstance(item, CandidateEvaluation):
                raise RepairCandidatePortfolioError(
                    "evaluations must contain CandidateEvaluation values",
                    reason_code=PortfolioReason.MALFORMED_INPUT,
                )
        object.__setattr__(self, "evaluations", evaluations)
        object.__setattr__(
            self,
            "ranked_admissible",
            _ids(self.ranked_admissible, "ranked_admissible", preserve_order=True),
        )
        object.__setattr__(
            self,
            "selected_candidate_id",
            _optional_text(
                self.selected_candidate_id, "selected_candidate_id", limit=MAX_ID_BYTES
            ),
        )
        object.__setattr__(
            self,
            "selection_identity",
            _optional_text(
                self.selection_identity, "selection_identity", limit=MAX_ID_BYTES
            ),
        )
        object.__setattr__(
            self,
            "replay_identity",
            _optional_text(self.replay_identity, "replay_identity", limit=MAX_ID_BYTES),
        )
        object.__setattr__(
            self,
            "flaky_lanes",
            _ids(self.flaky_lanes, "flaky_lanes", preserve_order=True),
        )
        object.__setattr__(
            self,
            "unavailable_lanes",
            _ids(self.unavailable_lanes, "unavailable_lanes", preserve_order=True),
        )
        object.__setattr__(
            self,
            "hard_obligation_ids",
            _ids(self.hard_obligation_ids, "hard_obligation_ids", preserve_order=True),
        )
        object.__setattr__(
            self, "seed", _nonneg_int(self.seed, "seed", maximum=2**63 - 1)
        )
        if not isinstance(self.budget, PortfolioSeedBudget):
            raise RepairCandidatePortfolioError(
                "budget must be PortfolioSeedBudget",
                reason_code=PortfolioReason.MALFORMED_INPUT,
            )
        object.__setattr__(
            self, "oracle_id", _optional_text(self.oracle_id, "oracle_id", limit=MAX_ID_BYTES)
        )

        if self.proposal_only is not True:
            raise PortfolioAuthorityError(
                "decisions must remain proposal-only",
                reason_code=PortfolioReason.PROPOSAL_ONLY,
            )
        object.__setattr__(self, "proposal_only", True)
        for name in (
            "write_authority",
            "semantic_authority",
            "grants_completion_authority",
            "weighted_authority_used",
        ):
            if getattr(self, name) is not False:
                raise PortfolioAuthorityError(
                    f"decision cannot claim {name}",
                    reason_code=PortfolioReason.AUTHORITY_CLAIM,
                )
            object.__setattr__(self, name, False)

        if self.disposition is PortfolioDisposition.SELECTED:
            if not self.selected_candidate_id:
                raise RepairCandidatePortfolioError(
                    "selected disposition requires selected_candidate_id",
                    reason_code=PortfolioReason.MALFORMED_INPUT,
                )
            if self.selected_candidate_id not in self.ranked_admissible:
                raise RepairCandidatePortfolioError(
                    "selected candidate must be hard-admissible",
                    reason_code=PortfolioReason.NO_HARD_ADMISSIBLE,
                )
        else:
            if self.selected_candidate_id:
                raise RepairCandidatePortfolioError(
                    "non-selected disposition cannot name a selected candidate",
                    reason_code=PortfolioReason.MALFORMED_INPUT,
                )

        object.__setattr__(
            self,
            "producer_id",
            _text(self.producer_id or PRODUCER_ID, "producer_id", limit=MAX_ID_BYTES),
        )
        object.__setattr__(
            self,
            "interface",
            _text(
                self.interface or REPAIR_CANDIDATE_DECISION_INTERFACE,
                "interface",
                limit=MAX_ID_BYTES,
            ),
        )

        # Derive identities when absent so sealed decisions are always addressable.
        if not self.selection_identity or not self.replay_identity:
            derived = derive_selection_replay_identities(self)
            if not self.selection_identity:
                object.__setattr__(self, "selection_identity", derived[0])
            if not self.replay_identity:
                object.__setattr__(self, "replay_identity", derived[1])

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "interface": self.interface,
            "disposition": self.disposition.value,
            "reason_codes": list(self.reason_codes),
            "evaluations": [item.to_dict() for item in self.evaluations],
            "ranked_admissible": list(self.ranked_admissible),
            "selected_candidate_id": self.selected_candidate_id,
            "selection_identity": self.selection_identity,
            "replay_identity": self.replay_identity,
            "flaky_lanes": list(self.flaky_lanes),
            "unavailable_lanes": list(self.unavailable_lanes),
            "hard_obligation_ids": list(self.hard_obligation_ids),
            "seed": self.seed,
            "budget": self.budget.to_dict(),
            "oracle_id": self.oracle_id,
            "proposal_only": True,
            "write_authority": False,
            "semantic_authority": False,
            "grants_completion_authority": False,
            "weighted_authority_used": False,
            "producer_id": self.producer_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RepairCandidateDecision":
        if not isinstance(payload, Mapping):
            raise RepairCandidatePortfolioError(
                "decision must be a mapping",
                reason_code=PortfolioReason.MALFORMED_INPUT,
            )
        if payload.get("schema") not in {None, "", PORTFOLIO_DECISION_SCHEMA}:
            raise RepairCandidatePortfolioError(
                "unsupported repair candidate decision schema",
                reason_code=PortfolioReason.MALFORMED_INPUT,
            )
        evaluations = tuple(
            CandidateEvaluation.from_dict(item)
            for item in (payload.get("evaluations") or ())
        )
        budget_raw = payload.get("budget")
        budget = (
            PortfolioSeedBudget.from_dict(budget_raw)
            if isinstance(budget_raw, Mapping) or budget_raw is None
            else PortfolioSeedBudget()
        )
        return cls(
            disposition=str(payload.get("disposition") or ""),
            reason_codes=tuple(payload.get("reason_codes") or ()),
            evaluations=evaluations,
            ranked_admissible=tuple(payload.get("ranked_admissible") or ()),
            selected_candidate_id=str(payload.get("selected_candidate_id") or ""),
            selection_identity=str(payload.get("selection_identity") or ""),
            replay_identity=str(payload.get("replay_identity") or ""),
            flaky_lanes=tuple(payload.get("flaky_lanes") or ()),
            unavailable_lanes=tuple(payload.get("unavailable_lanes") or ()),
            hard_obligation_ids=tuple(payload.get("hard_obligation_ids") or ()),
            seed=int(payload.get("seed") or DEFAULT_SEED),
            budget=budget,
            oracle_id=str(payload.get("oracle_id") or ""),
            producer_id=str(payload.get("producer_id") or PRODUCER_ID),
            interface=str(
                payload.get("interface") or REPAIR_CANDIDATE_DECISION_INTERFACE
            ),
        )


@dataclass(frozen=True)
class PortfolioRequest:
    """Frozen inputs for one multi-method portfolio evaluation."""

    candidates: tuple[PortfolioCandidate, ...]
    oracle: IndependentOracle
    hard_obligations: tuple[HardObligation, ...] = ()
    budget: PortfolioSeedBudget = field(default_factory=PortfolioSeedBudget)
    capability_support: Mapping[str, bool] = field(default_factory=dict)
    repository_tree_id: str = ""
    forest_id: str = ""
    policy_id: str = ""
    request_id: str = ""

    def __post_init__(self) -> None:
        candidates = tuple(self.candidates or ())
        if len(candidates) > MAX_CANDIDATES:
            raise PortfolioBoundsError(
                f"candidate count exceeds bound {MAX_CANDIDATES}",
                reason_code=PortfolioReason.BOUNDS_EXCEEDED,
            )
        seen: set[str] = set()
        for item in candidates:
            if not isinstance(item, PortfolioCandidate):
                raise RepairCandidatePortfolioError(
                    "candidates must be PortfolioCandidate values",
                    reason_code=PortfolioReason.MALFORMED_INPUT,
                )
            if item.candidate_id in seen:
                raise RepairCandidatePortfolioError(
                    "duplicate candidate_id in portfolio",
                    reason_code=PortfolioReason.DUPLICATE_CANDIDATE,
                )
            seen.add(item.candidate_id)
        object.__setattr__(self, "candidates", candidates)

        if not isinstance(self.oracle, IndependentOracle):
            raise RepairCandidatePortfolioError(
                "oracle must be IndependentOracle",
                reason_code=PortfolioReason.MALFORMED_INPUT,
            )
        if not self.oracle.is_independent():
            raise RepairCandidatePortfolioError(
                "oracle is not independent",
                reason_code=PortfolioReason.ORACLE_NOT_INDEPENDENT,
            )

        obligations = tuple(self.hard_obligations or ())
        if len(obligations) > MAX_OBLIGATIONS:
            raise PortfolioBoundsError(
                f"hard obligation count exceeds bound {MAX_OBLIGATIONS}",
                reason_code=PortfolioReason.BOUNDS_EXCEEDED,
            )
        for item in obligations:
            if not isinstance(item, HardObligation):
                raise RepairCandidatePortfolioError(
                    "hard_obligations must be HardObligation values",
                    reason_code=PortfolioReason.MALFORMED_INPUT,
                )
        object.__setattr__(self, "hard_obligations", obligations)

        if not isinstance(self.budget, PortfolioSeedBudget):
            raise RepairCandidatePortfolioError(
                "budget must be PortfolioSeedBudget",
                reason_code=PortfolioReason.MALFORMED_INPUT,
            )
        object.__setattr__(
            self, "capability_support", _mapping_proxy(self.capability_support, "capability_support")
        )
        object.__setattr__(
            self,
            "repository_tree_id",
            _optional_text(self.repository_tree_id, "repository_tree_id", limit=MAX_ID_BYTES),
        )
        object.__setattr__(
            self, "forest_id", _optional_text(self.forest_id, "forest_id", limit=MAX_ID_BYTES)
        )
        object.__setattr__(
            self, "policy_id", _optional_text(self.policy_id, "policy_id", limit=MAX_ID_BYTES)
        )
        object.__setattr__(
            self,
            "request_id",
            _optional_text(self.request_id, "request_id", limit=MAX_ID_BYTES),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PORTFOLIO_REQUEST_SCHEMA,
            "contract_version": CONTRACT_VERSION,
            "candidates": [item.to_dict() for item in self.candidates],
            "oracle": self.oracle.to_dict(),
            "hard_obligations": [item.to_dict() for item in self.hard_obligations],
            "budget": self.budget.to_dict(),
            "capability_support": dict(self.capability_support),
            "repository_tree_id": self.repository_tree_id,
            "forest_id": self.forest_id,
            "policy_id": self.policy_id,
            "request_id": self.request_id,
        }

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict())


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def derive_selection_replay_identities(
    decision: RepairCandidateDecision | Mapping[str, Any],
) -> tuple[str, str]:
    """Derive content-addressed selection and replay identities.

    Selection identity binds the chosen candidate (or abstention) and ranking.
    Replay identity binds the full sealed decision body that must match on
    re-evaluation of the same frozen inputs.
    """

    if isinstance(decision, RepairCandidateDecision):
        payload = decision.to_dict()
    elif isinstance(decision, Mapping):
        payload = dict(decision)
    else:
        raise RepairCandidatePortfolioError(
            "decision must be RepairCandidateDecision or mapping",
            reason_code=PortfolioReason.MALFORMED_INPUT,
        )

    selection_body = {
        "kind": "repair_candidate_selection",
        "interface": REPAIR_CANDIDATE_DECISION_INTERFACE,
        "disposition": payload.get("disposition"),
        "selected_candidate_id": payload.get("selected_candidate_id") or "",
        "ranked_admissible": list(payload.get("ranked_admissible") or ()),
        "reason_codes": list(payload.get("reason_codes") or ()),
        "hard_obligation_ids": list(payload.get("hard_obligation_ids") or ()),
        "seed": payload.get("seed"),
        "oracle_id": payload.get("oracle_id") or "",
        "budget": payload.get("budget") or {},
        # Evaluations without soft scores: hard admissibility only.
        "hard_admissible": [
            {
                "candidate_id": item.get("candidate_id"),
                "hard_admissible": item.get("hard_admissible"),
                "blast_radius": item.get("blast_radius"),
                "resource_cost": item.get("resource_cost"),
                "hard_failures": list(item.get("hard_failures") or ()),
                "rejection_reasons": list(item.get("rejection_reasons") or ()),
            }
            for item in (payload.get("evaluations") or ())
        ],
    }
    selection_identity = content_identity(selection_body)

    replay_body = {
        "kind": "repair_candidate_replay",
        "interface": REPAIR_CANDIDATE_DECISION_INTERFACE,
        "selection_identity": selection_identity,
        "disposition": payload.get("disposition"),
        "selected_candidate_id": payload.get("selected_candidate_id") or "",
        "ranked_admissible": list(payload.get("ranked_admissible") or ()),
        "reason_codes": list(payload.get("reason_codes") or ()),
        "evaluations": list(payload.get("evaluations") or ()),
        "flaky_lanes": list(payload.get("flaky_lanes") or ()),
        "unavailable_lanes": list(payload.get("unavailable_lanes") or ()),
        "hard_obligation_ids": list(payload.get("hard_obligation_ids") or ()),
        "seed": payload.get("seed"),
        "budget": payload.get("budget") or {},
        "oracle_id": payload.get("oracle_id") or "",
        "producer_id": payload.get("producer_id") or PRODUCER_ID,
    }
    replay_identity = content_identity(replay_body)
    return selection_identity, replay_identity


def prove_selection_replay_identity(
    first: RepairCandidateDecision,
    second: RepairCandidateDecision,
) -> bool:
    """Return True when two decisions prove selection and replay identity."""

    if not isinstance(first, RepairCandidateDecision) or not isinstance(
        second, RepairCandidateDecision
    ):
        raise RepairCandidatePortfolioError(
            "both arguments must be RepairCandidateDecision",
            reason_code=PortfolioReason.MALFORMED_INPUT,
        )
    if first.selection_identity != second.selection_identity:
        return False
    if first.replay_identity != second.replay_identity:
        return False
    if first.disposition != second.disposition:
        return False
    if first.selected_candidate_id != second.selected_candidate_id:
        return False
    if first.ranked_admissible != second.ranked_admissible:
        return False
    # Recompute from sealed payloads to catch forged identity fields.
    first_sel, first_rep = derive_selection_replay_identities(first)
    second_sel, second_rep = derive_selection_replay_identities(second)
    if first.selection_identity != first_sel or first.replay_identity != first_rep:
        return False
    if second.selection_identity != second_sel or second.replay_identity != second_rep:
        return False
    return first_sel == second_sel and first_rep == second_rep


# ---------------------------------------------------------------------------
# Lane evaluation
# ---------------------------------------------------------------------------


def default_hard_obligations() -> tuple[HardObligation, ...]:
    """Default hard obligations covering the full independent multi-method set."""

    return tuple(
        HardObligation(
            obligation_id=f"hard:{lane.value}",
            lane=lane,
            required=True,
            description=f"Required independent {lane.value} gate",
        )
        for lane in sorted(DEFAULT_HARD_LANES, key=lambda item: item.value)
    )


def _lane_budget_cap(lane: PortfolioLane, budget: PortfolioSeedBudget) -> int:
    if lane is PortfolioLane.PROPERTY_BASED:
        return budget.max_property_cases
    if lane is PortfolioLane.FUZZ:
        return budget.max_fuzz_inputs
    if lane is PortfolioLane.CONCOLIC:
        return budget.max_concolic_paths
    if lane is PortfolioLane.MUTATION:
        return budget.max_mutation_ops
    return budget.max_wall_ms


def _resolve_hard_lanes(
    obligations: Sequence[HardObligation],
) -> frozenset[PortfolioLane]:
    required = {item.lane for item in obligations if item.required}
    if not required:
        return DEFAULT_HARD_LANES
    return frozenset(required)


def evaluate_lane(
    *,
    lane: PortfolioLane,
    candidate: PortfolioCandidate,
    oracle: IndependentOracle,
    budget: PortfolioSeedBudget,
    hard: bool,
    capability_support: Mapping[str, bool],
) -> LaneResult:
    """Evaluate one validation lane under fixed seed/budget (deterministic)."""

    seed = budget.seed
    cap = _lane_budget_cap(lane, budget)
    env_supported = capability_support.get(lane.value, True)

    # Capability gate for optional lanes.
    if not env_supported:
        outcome = LaneOutcome.UNAVAILABLE
        reasons = (PortfolioReason.SOFT_DEBT_ONLY.value,)
        if hard:
            reasons = (PortfolioReason.HARD_OBLIGATION_UNAVAILABLE.value,)
        evidence = content_identity(
            {
                "lane": lane.value,
                "candidate_id": candidate.candidate_id,
                "outcome": outcome.value,
                "seed": seed,
                "reason": "capability_unavailable",
            }
        )
        return LaneResult(
            lane=lane,
            outcome=outcome,
            hard=hard,
            seed=seed,
            budget_used=0,
            cases_run=0,
            evidence_id=evidence,
            reason_codes=reasons,
        )

    obs = candidate.observation_for(lane)

    # Self-authored tests are never admissible evidence.
    if obs.uses_self_authored_tests:
        evidence = content_identity(
            {
                "lane": lane.value,
                "candidate_id": candidate.candidate_id,
                "outcome": LaneOutcome.FAIL.value,
                "seed": seed,
                "reason": PortfolioReason.SELF_AUTHORED_TEST.value,
            }
        )
        return LaneResult(
            lane=lane,
            outcome=LaneOutcome.FAIL,
            hard=hard,
            seed=seed,
            budget_used=min(obs.budget_used, cap),
            cases_run=min(obs.cases_run, cap),
            evidence_id=evidence,
            reason_codes=(PortfolioReason.SELF_AUTHORED_TEST.value,),
            oracle_id=obs.oracle_id,
        )

    # Candidate-as-oracle / self-validation.
    if obs.candidate_claims_oracle:
        evidence = content_identity(
            {
                "lane": lane.value,
                "candidate_id": candidate.candidate_id,
                "outcome": LaneOutcome.FAIL.value,
                "seed": seed,
                "reason": PortfolioReason.CANDIDATE_AS_ORACLE.value,
            }
        )
        return LaneResult(
            lane=lane,
            outcome=LaneOutcome.FAIL,
            hard=hard,
            seed=seed,
            budget_used=min(obs.budget_used, cap),
            cases_run=min(obs.cases_run, cap),
            evidence_id=evidence,
            reason_codes=(PortfolioReason.CANDIDATE_AS_ORACLE.value,),
            oracle_id=obs.oracle_id,
        )

    if candidate.claimed_oracle_ids:
        claimed = set(candidate.claimed_oracle_ids)
        if oracle.oracle_id in claimed or any(
            _normalize_oracle_source(item) in _FORBIDDEN_ORACLE_SOURCES
            for item in candidate.claimed_oracle_ids
        ):
            evidence = content_identity(
                {
                    "lane": lane.value,
                    "candidate_id": candidate.candidate_id,
                    "outcome": LaneOutcome.FAIL.value,
                    "seed": seed,
                    "reason": PortfolioReason.CANDIDATE_AS_ORACLE.value,
                }
            )
            return LaneResult(
                lane=lane,
                outcome=LaneOutcome.FAIL,
                hard=hard,
                seed=seed,
                budget_used=min(obs.budget_used, cap),
                cases_run=min(obs.cases_run, cap),
                evidence_id=evidence,
                reason_codes=(PortfolioReason.CANDIDATE_AS_ORACLE.value,),
                oracle_id=obs.oracle_id,
            )

    # Mutation (and any lane that names an oracle) must bind the independent oracle.
    if lane is PortfolioLane.MUTATION:
        if not obs.oracle_id:
            evidence = content_identity(
                {
                    "lane": lane.value,
                    "candidate_id": candidate.candidate_id,
                    "outcome": LaneOutcome.FAIL.value,
                    "seed": seed,
                    "reason": PortfolioReason.MISSING_ORACLE.value,
                }
            )
            return LaneResult(
                lane=lane,
                outcome=LaneOutcome.FAIL,
                hard=hard,
                seed=seed,
                budget_used=min(obs.budget_used, cap),
                cases_run=min(obs.cases_run, cap),
                evidence_id=evidence,
                reason_codes=(PortfolioReason.MISSING_ORACLE.value,),
            )
        if obs.oracle_id != oracle.oracle_id:
            evidence = content_identity(
                {
                    "lane": lane.value,
                    "candidate_id": candidate.candidate_id,
                    "outcome": LaneOutcome.FAIL.value,
                    "seed": seed,
                    "reason": PortfolioReason.ORACLE_NOT_INDEPENDENT.value,
                    "observed_oracle": obs.oracle_id,
                    "required_oracle": oracle.oracle_id,
                }
            )
            return LaneResult(
                lane=lane,
                outcome=LaneOutcome.FAIL,
                hard=hard,
                seed=seed,
                budget_used=min(obs.budget_used, cap),
                cases_run=min(obs.cases_run, cap),
                evidence_id=evidence,
                reason_codes=(PortfolioReason.ORACLE_NOT_INDEPENDENT.value,),
                oracle_id=obs.oracle_id,
            )

    # Candidate-authored tests overlapping oracle tests → reject.
    if candidate.authored_test_ids and oracle.test_ids:
        overlap = set(candidate.authored_test_ids).intersection(oracle.test_ids)
        if overlap:
            evidence = content_identity(
                {
                    "lane": lane.value,
                    "candidate_id": candidate.candidate_id,
                    "outcome": LaneOutcome.FAIL.value,
                    "seed": seed,
                    "reason": PortfolioReason.SELF_AUTHORED_TEST.value,
                    "overlap": sorted(overlap),
                }
            )
            return LaneResult(
                lane=lane,
                outcome=LaneOutcome.FAIL,
                hard=hard,
                seed=seed,
                budget_used=min(obs.budget_used, cap),
                cases_run=min(obs.cases_run, cap),
                evidence_id=evidence,
                reason_codes=(PortfolioReason.SELF_AUTHORED_TEST.value,),
                oracle_id=obs.oracle_id,
            )

    if not obs.supported:
        outcome = LaneOutcome.UNAVAILABLE
        reasons = list(obs.reason_codes) or (
            [PortfolioReason.HARD_OBLIGATION_UNAVAILABLE.value]
            if hard
            else [PortfolioReason.SOFT_DEBT_ONLY.value]
        )
        evidence = content_identity(
            {
                "lane": lane.value,
                "candidate_id": candidate.candidate_id,
                "outcome": outcome.value,
                "seed": seed,
                "reason": "lane_unsupported",
            }
        )
        return LaneResult(
            lane=lane,
            outcome=outcome,
            hard=hard,
            seed=seed,
            budget_used=min(obs.budget_used, cap),
            cases_run=min(obs.cases_run, cap),
            evidence_id=evidence,
            reason_codes=tuple(reasons),
            oracle_id=obs.oracle_id,
        )

    # Budget exceed → fail-closed for hard, unavailable/soft debt for optional.
    if obs.budget_used > cap or obs.cases_run > cap:
        outcome = LaneOutcome.FAIL if hard else LaneOutcome.UNAVAILABLE
        evidence = content_identity(
            {
                "lane": lane.value,
                "candidate_id": candidate.candidate_id,
                "outcome": outcome.value,
                "seed": seed,
                "reason": PortfolioReason.BUDGET_EXCEEDED.value,
                "budget_used": obs.budget_used,
                "cap": cap,
            }
        )
        return LaneResult(
            lane=lane,
            outcome=outcome,
            hard=hard,
            seed=seed,
            budget_used=obs.budget_used,
            cases_run=obs.cases_run,
            evidence_id=evidence,
            reason_codes=(PortfolioReason.BUDGET_EXCEEDED.value,),
            oracle_id=obs.oracle_id,
        )

    status = obs.status.casefold()
    if status in {"pass", "passed", "ok", "success"}:
        outcome = LaneOutcome.PASS
        reasons = list(obs.reason_codes) or (PortfolioReason.OK.value,)
    elif status in {"fail", "failed", "error", "reject", "rejected"}:
        outcome = LaneOutcome.FAIL
        reasons = list(obs.reason_codes) or (PortfolioReason.LANE_FAIL.value,)
    elif status in {"flaky", "intermittent", "non_deterministic"}:
        outcome = LaneOutcome.FLAKY
        reasons = list(obs.reason_codes) or (
            PortfolioReason.HARD_OBLIGATION_FLAKY.value
            if hard
            else ("flaky_lane",)
        )
    elif status in {"unavailable", "unsupported", "missing"}:
        outcome = LaneOutcome.UNAVAILABLE
        reasons = list(obs.reason_codes) or (
            PortfolioReason.HARD_OBLIGATION_UNAVAILABLE.value
            if hard
            else [PortfolioReason.SOFT_DEBT_ONLY.value]
        )
    elif status in {"skipped", "skip"}:
        outcome = LaneOutcome.SKIPPED
        reasons = list(obs.reason_codes) or ("skipped",)
    else:
        # Unknown status → fail closed.
        outcome = LaneOutcome.FAIL
        reasons = list(obs.reason_codes) or (PortfolioReason.MALFORMED_INPUT.value,)

    # Soft scores never upgrade a non-pass outcome (hard failures not averaged).
    if outcome is not LaneOutcome.PASS and obs.soft_score is not None:
        extra = PortfolioReason.HARD_FAILURE_NOT_AVERAGED.value
        if extra not in reasons:
            reasons = list(reasons) + [extra]

    evidence = content_identity(
        {
            "lane": lane.value,
            "candidate_id": candidate.candidate_id,
            "outcome": outcome.value,
            "seed": seed,
            "cases_run": min(obs.cases_run, cap),
            "budget_used": min(obs.budget_used, cap),
            "evidence_refs": list(obs.evidence_refs),
            "oracle_id": obs.oracle_id or oracle.oracle_id,
            # soft_score intentionally excluded from evidence authority binding
        }
    )
    return LaneResult(
        lane=lane,
        outcome=outcome,
        hard=hard,
        seed=seed,
        budget_used=min(obs.budget_used, cap),
        cases_run=min(obs.cases_run, cap),
        evidence_id=evidence,
        reason_codes=tuple(reasons),
        oracle_id=obs.oracle_id or (
            oracle.oracle_id if lane is PortfolioLane.MUTATION else ""
        ),
    )


def evaluate_candidate(
    candidate: PortfolioCandidate,
    *,
    oracle: IndependentOracle,
    hard_obligations: Sequence[HardObligation],
    budget: PortfolioSeedBudget,
    capability_support: Mapping[str, bool],
) -> CandidateEvaluation:
    """Run the full multi-method lane set for one candidate."""

    hard_lanes = _resolve_hard_lanes(hard_obligations)
    # Resource-cost budget at candidate level.
    if candidate.resource_cost > budget.max_resource_cost:
        return CandidateEvaluation(
            candidate_id=candidate.candidate_id,
            hard_admissible=False,
            blast_radius=candidate.blast_radius,
            resource_cost=candidate.resource_cost,
            lane_results=(),
            hard_failures=(PortfolioReason.BUDGET_EXCEEDED.value,),
            rejection_reasons=(PortfolioReason.BUDGET_EXCEEDED.value,),
            ranking_key=(
                candidate.blast_radius,
                candidate.resource_cost,
                candidate.candidate_id,
            ),
        )

    results: list[LaneResult] = []
    for lane in PORTFOLIO_LANE_ORDER:
        hard = lane in hard_lanes
        results.append(
            evaluate_lane(
                lane=lane,
                candidate=candidate,
                oracle=oracle,
                budget=budget,
                hard=hard,
                capability_support=capability_support,
            )
        )

    hard_failures: list[str] = []
    flaky: list[str] = []
    unavailable: list[str] = []
    rejection: list[str] = []
    soft_debt: list[str] = []

    for result in results:
        if result.outcome is LaneOutcome.FLAKY:
            flaky.append(result.lane.value)
        if result.outcome is LaneOutcome.UNAVAILABLE:
            unavailable.append(result.lane.value)

        if result.hard:
            if result.outcome is LaneOutcome.PASS:
                continue
            if result.outcome is LaneOutcome.FAIL:
                hard_failures.append(result.lane.value)
                for code in result.reason_codes:
                    if code not in rejection:
                        rejection.append(code)
                if PortfolioReason.HARD_OBLIGATION_FAILED.value not in rejection:
                    # Prefer specific codes already present; still tag hard fail.
                    if not any(
                        code
                        in {
                            PortfolioReason.SELF_AUTHORED_TEST.value,
                            PortfolioReason.CANDIDATE_AS_ORACLE.value,
                            PortfolioReason.ORACLE_NOT_INDEPENDENT.value,
                            PortfolioReason.MISSING_ORACLE.value,
                            PortfolioReason.BUDGET_EXCEEDED.value,
                        }
                        for code in result.reason_codes
                    ):
                        rejection.append(PortfolioReason.HARD_OBLIGATION_FAILED.value)
            elif result.outcome is LaneOutcome.FLAKY:
                hard_failures.append(result.lane.value)
                if PortfolioReason.HARD_OBLIGATION_FLAKY.value not in rejection:
                    rejection.append(PortfolioReason.HARD_OBLIGATION_FLAKY.value)
            elif result.outcome is LaneOutcome.UNAVAILABLE:
                hard_failures.append(result.lane.value)
                if PortfolioReason.HARD_OBLIGATION_UNAVAILABLE.value not in rejection:
                    rejection.append(PortfolioReason.HARD_OBLIGATION_UNAVAILABLE.value)
            elif result.outcome is LaneOutcome.SKIPPED:
                hard_failures.append(result.lane.value)
                if PortfolioReason.HARD_OBLIGATION_UNAVAILABLE.value not in rejection:
                    rejection.append(PortfolioReason.HARD_OBLIGATION_UNAVAILABLE.value)
        else:
            if result.outcome in {
                LaneOutcome.UNAVAILABLE,
                LaneOutcome.FLAKY,
                LaneOutcome.SKIPPED,
            }:
                soft_debt.append(result.lane.value)
            elif result.outcome is LaneOutcome.FAIL:
                # Optional lane fail becomes soft debt unless explicitly hard.
                soft_debt.append(result.lane.value)

    # Global candidate-level self-authored / oracle claims (even if no lane flagged).
    if candidate.authored_test_ids and oracle.test_ids:
        if set(candidate.authored_test_ids).intersection(oracle.test_ids):
            if PortfolioReason.SELF_AUTHORED_TEST.value not in rejection:
                rejection.append(PortfolioReason.SELF_AUTHORED_TEST.value)
            hard_failures.append("oracle_tests")

    hard_admissible = not hard_failures and not any(
        code
        in {
            PortfolioReason.SELF_AUTHORED_TEST.value,
            PortfolioReason.CANDIDATE_AS_ORACLE.value,
            PortfolioReason.ORACLE_NOT_INDEPENDENT.value,
        }
        for code in rejection
    )

    return CandidateEvaluation(
        candidate_id=candidate.candidate_id,
        hard_admissible=hard_admissible,
        blast_radius=candidate.blast_radius,
        resource_cost=candidate.resource_cost,
        lane_results=tuple(results),
        hard_failures=tuple(hard_failures),
        flaky_lanes=tuple(flaky),
        unavailable_lanes=tuple(unavailable),
        rejection_reasons=tuple(rejection),
        soft_debt=tuple(soft_debt),
        ranking_key=(
            candidate.blast_radius,
            candidate.resource_cost,
            candidate.candidate_id,
        ),
    )


def rank_hard_admissible(
    evaluations: Sequence[CandidateEvaluation],
) -> tuple[CandidateEvaluation, ...]:
    """Rank only hard-admissible candidates by blast radius, then resource cost."""

    admissible = [item for item in evaluations if item.hard_admissible]
    admissible.sort(
        key=lambda item: (item.blast_radius, item.resource_cost, item.candidate_id)
    )
    return tuple(admissible)


# ---------------------------------------------------------------------------
# Portfolio service
# ---------------------------------------------------------------------------


class LaneRunner(Protocol):
    """Optional injectable lane runner for hermetic / external tooling."""

    def __call__(
        self,
        *,
        lane: PortfolioLane,
        candidate: PortfolioCandidate,
        oracle: IndependentOracle,
        budget: PortfolioSeedBudget,
        hard: bool,
        capability_support: Mapping[str, bool],
    ) -> LaneResult: ...


@dataclass
class RepairCandidatePortfolio:
    """Independent multi-method repair candidate portfolio selector.

    Interface: ``RepairCandidatePortfolio@1``
    """

    INTERFACE: ClassVar[str] = REPAIR_CANDIDATE_PORTFOLIO_INTERFACE
    VERSION: ClassVar[str] = REPAIR_CANDIDATE_PORTFOLIO_VERSION

    lane_runner: Callable[..., LaneResult] | None = None
    producer_id: str = PRODUCER_ID

    def evaluate(self, request: PortfolioRequest) -> RepairCandidateDecision:
        """Evaluate all candidates and emit a sealed portfolio decision."""

        if not isinstance(request, PortfolioRequest):
            raise RepairCandidatePortfolioError(
                "request must be PortfolioRequest",
                reason_code=PortfolioReason.MALFORMED_INPUT,
            )
        if not request.candidates:
            decision = RepairCandidateDecision(
                disposition=PortfolioDisposition.ABSTAIN,
                reason_codes=(
                    PortfolioReason.NO_CANDIDATES.value,
                    PortfolioReason.CORRECT_ABSTENTION.value,
                ),
                evaluations=(),
                ranked_admissible=(),
                selected_candidate_id="",
                flaky_lanes=(),
                unavailable_lanes=(),
                hard_obligation_ids=tuple(
                    item.obligation_id for item in request.hard_obligations
                ),
                seed=request.budget.seed,
                budget=request.budget,
                oracle_id=request.oracle.oracle_id,
                producer_id=self.producer_id,
            )
            return decision

        obligations = request.hard_obligations or default_hard_obligations()
        hard_lanes = _resolve_hard_lanes(obligations)
        runner = self.lane_runner or evaluate_lane

        evaluations: list[CandidateEvaluation] = []
        for candidate in request.candidates:
            # Re-evaluate via runner for each lane so injectors work.
            if self.lane_runner is None:
                evaluation = evaluate_candidate(
                    candidate,
                    oracle=request.oracle,
                    hard_obligations=obligations,
                    budget=request.budget,
                    capability_support=request.capability_support,
                )
            else:
                results: list[LaneResult] = []
                for lane in PORTFOLIO_LANE_ORDER:
                    results.append(
                        runner(
                            lane=lane,
                            candidate=candidate,
                            oracle=request.oracle,
                            budget=request.budget,
                            hard=lane in hard_lanes,
                            capability_support=request.capability_support,
                        )
                    )
                evaluation = _aggregate_results(
                    candidate=candidate,
                    results=tuple(results),
                    oracle=request.oracle,
                )
            evaluations.append(evaluation)

        ranked = rank_hard_admissible(evaluations)
        ranked_ids = tuple(item.candidate_id for item in ranked)

        all_flaky: list[str] = []
        all_unavailable: list[str] = []
        for evaluation in evaluations:
            for lane_name in evaluation.flaky_lanes:
                key = f"{evaluation.candidate_id}:{lane_name}"
                if key not in all_flaky:
                    all_flaky.append(key)
            for lane_name in evaluation.unavailable_lanes:
                key = f"{evaluation.candidate_id}:{lane_name}"
                if key not in all_unavailable:
                    all_unavailable.append(key)

        obligation_ids = tuple(item.obligation_id for item in obligations)

        if ranked:
            selected = ranked[0]
            reasons = [
                PortfolioReason.OK.value,
                PortfolioReason.ALL_HARD_OBLIGATIONS_MET.value,
                PortfolioReason.MINIMAL_BLAST_RADIUS.value,
            ]
            if len(ranked) > 1 and (
                selected.resource_cost < ranked[1].resource_cost
                or selected.blast_radius < ranked[1].blast_radius
            ):
                reasons.append(PortfolioReason.MINIMAL_RESOURCE_COST.value)
            decision = RepairCandidateDecision(
                disposition=PortfolioDisposition.SELECTED,
                reason_codes=tuple(reasons),
                evaluations=tuple(evaluations),
                ranked_admissible=ranked_ids,
                selected_candidate_id=selected.candidate_id,
                flaky_lanes=tuple(all_flaky),
                unavailable_lanes=tuple(all_unavailable),
                hard_obligation_ids=obligation_ids,
                seed=request.budget.seed,
                budget=request.budget,
                oracle_id=request.oracle.oracle_id,
                producer_id=self.producer_id,
            )
            return decision

        # Correct abstention: no hard-admissible candidates.
        reject_codes: list[str] = []
        for evaluation in evaluations:
            for code in evaluation.rejection_reasons:
                if code not in reject_codes:
                    reject_codes.append(code)
        reasons = [
            PortfolioReason.NO_HARD_ADMISSIBLE.value,
            PortfolioReason.CORRECT_ABSTENTION.value,
            *reject_codes,
        ]
        # Distinct reject vs abstain: reject only when every candidate was
        # positively refuted (hard fail); abstain when empty admissible set
        # may include capability/uncertainty debt. Prefer abstain (fail closed
        # without false selection) — callers may treat reject_codes for audit.
        disposition = PortfolioDisposition.ABSTAIN
        if evaluations and all(
            evaluation.hard_failures and not evaluation.hard_admissible
            for evaluation in evaluations
        ):
            # Still abstain to preserve correct abstention semantics unless
            # every candidate is hard-failed with explicit refutation codes.
            if any(
                PortfolioReason.SELF_AUTHORED_TEST.value in evaluation.rejection_reasons
                or PortfolioReason.CANDIDATE_AS_ORACLE.value
                in evaluation.rejection_reasons
                for evaluation in evaluations
            ) and all(not evaluation.hard_admissible for evaluation in evaluations):
                disposition = PortfolioDisposition.REJECT
                if PortfolioReason.CORRECT_ABSTENTION.value in reasons:
                    reasons = [
                        code
                        for code in reasons
                        if code != PortfolioReason.CORRECT_ABSTENTION.value
                    ]

        decision = RepairCandidateDecision(
            disposition=disposition,
            reason_codes=tuple(reasons),
            evaluations=tuple(evaluations),
            ranked_admissible=(),
            selected_candidate_id="",
            flaky_lanes=tuple(all_flaky),
            unavailable_lanes=tuple(all_unavailable),
            hard_obligation_ids=obligation_ids,
            seed=request.budget.seed,
            budget=request.budget,
            oracle_id=request.oracle.oracle_id,
            producer_id=self.producer_id,
        )
        return decision

    def select(self, request: PortfolioRequest) -> RepairCandidateDecision:
        """Alias for :meth:`evaluate`."""

        return self.evaluate(request)


def _aggregate_results(
    *,
    candidate: PortfolioCandidate,
    results: Sequence[LaneResult],
    oracle: IndependentOracle,
) -> CandidateEvaluation:
    hard_failures: list[str] = []
    flaky: list[str] = []
    unavailable: list[str] = []
    rejection: list[str] = []
    soft_debt: list[str] = []

    for result in results:
        if result.outcome is LaneOutcome.FLAKY:
            flaky.append(result.lane.value)
        if result.outcome is LaneOutcome.UNAVAILABLE:
            unavailable.append(result.lane.value)
        if result.hard:
            if result.outcome is not LaneOutcome.PASS:
                hard_failures.append(result.lane.value)
                for code in result.reason_codes:
                    if code not in rejection:
                        rejection.append(code)
                if result.outcome is LaneOutcome.FAIL:
                    if PortfolioReason.HARD_OBLIGATION_FAILED.value not in rejection:
                        if not any(
                            code
                            in {
                                PortfolioReason.SELF_AUTHORED_TEST.value,
                                PortfolioReason.CANDIDATE_AS_ORACLE.value,
                                PortfolioReason.ORACLE_NOT_INDEPENDENT.value,
                                PortfolioReason.MISSING_ORACLE.value,
                                PortfolioReason.BUDGET_EXCEEDED.value,
                            }
                            for code in result.reason_codes
                        ):
                            rejection.append(
                                PortfolioReason.HARD_OBLIGATION_FAILED.value
                            )
                elif result.outcome is LaneOutcome.FLAKY:
                    if PortfolioReason.HARD_OBLIGATION_FLAKY.value not in rejection:
                        rejection.append(PortfolioReason.HARD_OBLIGATION_FLAKY.value)
                elif result.outcome in {
                    LaneOutcome.UNAVAILABLE,
                    LaneOutcome.SKIPPED,
                }:
                    if (
                        PortfolioReason.HARD_OBLIGATION_UNAVAILABLE.value
                        not in rejection
                    ):
                        rejection.append(
                            PortfolioReason.HARD_OBLIGATION_UNAVAILABLE.value
                        )
        else:
            if result.outcome is not LaneOutcome.PASS:
                soft_debt.append(result.lane.value)

    if candidate.authored_test_ids and oracle.test_ids:
        if set(candidate.authored_test_ids).intersection(oracle.test_ids):
            if PortfolioReason.SELF_AUTHORED_TEST.value not in rejection:
                rejection.append(PortfolioReason.SELF_AUTHORED_TEST.value)
            hard_failures.append("oracle_tests")

    hard_admissible = not hard_failures and not any(
        code
        in {
            PortfolioReason.SELF_AUTHORED_TEST.value,
            PortfolioReason.CANDIDATE_AS_ORACLE.value,
            PortfolioReason.ORACLE_NOT_INDEPENDENT.value,
        }
        for code in rejection
    )
    return CandidateEvaluation(
        candidate_id=candidate.candidate_id,
        hard_admissible=hard_admissible,
        blast_radius=candidate.blast_radius,
        resource_cost=candidate.resource_cost,
        lane_results=tuple(results),
        hard_failures=tuple(hard_failures),
        flaky_lanes=tuple(flaky),
        unavailable_lanes=tuple(unavailable),
        rejection_reasons=tuple(rejection),
        soft_debt=tuple(soft_debt),
        ranking_key=(
            candidate.blast_radius,
            candidate.resource_cost,
            candidate.candidate_id,
        ),
    )


def create_repair_candidate_portfolio(
    *,
    lane_runner: Callable[..., LaneResult] | None = None,
    producer_id: str = PRODUCER_ID,
) -> RepairCandidatePortfolio:
    """Factory for :class:`RepairCandidatePortfolio`."""

    return RepairCandidatePortfolio(lane_runner=lane_runner, producer_id=producer_id)


def select_repair_candidate(request: PortfolioRequest) -> RepairCandidateDecision:
    """Module-level convenience for the default portfolio selector."""

    return create_repair_candidate_portfolio().evaluate(request)


def evaluate_repair_candidate_portfolio(
    request: PortfolioRequest,
) -> RepairCandidateDecision:
    """Module-level alias matching plan wording."""

    return select_repair_candidate(request)


# ---------------------------------------------------------------------------
# Passing-observation helpers (fixtures / hermetic harnesses)
# ---------------------------------------------------------------------------


def passing_observations(
    *,
    oracle_id: str,
    optional_status: str = "pass",
    hard_status: str = "pass",
    cases: int = 8,
) -> dict[str, dict[str, Any]]:
    """Build lane observations where all lanes report a uniform status."""

    out: dict[str, dict[str, Any]] = {}
    for lane in PORTFOLIO_LANE_ORDER:
        hard = lane in DEFAULT_HARD_LANES
        status = hard_status if hard else optional_status
        payload: dict[str, Any] = {
            "supported": True,
            "status": status,
            "cases_run": cases,
            "budget_used": cases,
            "uses_self_authored_tests": False,
            "candidate_claims_oracle": False,
            "evidence_refs": (f"evidence:{lane.value}",),
            "reason_codes": (PortfolioReason.OK.value,),
        }
        if lane is PortfolioLane.MUTATION:
            payload["oracle_id"] = oracle_id
        out[lane.value] = payload
    return out


def all_lanes_supported() -> dict[str, bool]:
    return {lane.value: True for lane in PORTFOLIO_LANE_ORDER}


__all__ = (
    "REPAIR_CANDIDATE_PORTFOLIO_INTERFACE",
    "REPAIR_CANDIDATE_DECISION_INTERFACE",
    "REPAIR_CANDIDATE_PORTFOLIO_VERSION",
    "PORTFOLIO_REQUEST_SCHEMA",
    "PORTFOLIO_CANDIDATE_SCHEMA",
    "PORTFOLIO_DECISION_SCHEMA",
    "PORTFOLIO_LANE_ORDER",
    "DEFAULT_HARD_LANES",
    "OPTIONAL_CAPABILITY_LANES",
    "PRODUCER_ID",
    "DEFAULT_SEED",
    "RepairCandidatePortfolioError",
    "PortfolioAuthorityError",
    "PortfolioBoundsError",
    "PortfolioLane",
    "LaneOutcome",
    "PortfolioDisposition",
    "PortfolioReason",
    "PortfolioSeedBudget",
    "IndependentOracle",
    "HardObligation",
    "LaneObservation",
    "PortfolioCandidate",
    "LaneResult",
    "CandidateEvaluation",
    "RepairCandidateDecision",
    "PortfolioRequest",
    "RepairCandidatePortfolio",
    "create_repair_candidate_portfolio",
    "select_repair_candidate",
    "evaluate_repair_candidate_portfolio",
    "evaluate_lane",
    "evaluate_candidate",
    "rank_hard_admissible",
    "default_hard_obligations",
    "derive_selection_replay_identities",
    "prove_selection_replay_identity",
    "passing_observations",
    "all_lanes_supported",
)
