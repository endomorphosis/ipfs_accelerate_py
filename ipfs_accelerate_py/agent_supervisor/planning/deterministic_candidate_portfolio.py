"""DCR-062: generate and admit a finite symbolic candidate portfolio.

Interfaces
----------
* ``RepairCandidate@1`` — one body-free repair candidate bound to current
  evidence and an exact operator CID.
* ``CandidateAdmission@1`` — unique-admission decision over a finite portfolio.

Normative rules (fail-closed)
-----------------------------
* Enumerate only registered operators with bounded, body-free arguments.
* Rank by proved applicability, risk, edit size, resource cost, and
  validation strength (fixed-point integer score terms).
* Admit only when exactly one eligible candidate is uniquely best.
* Ties and unknowns abstain; never invent a winner.
* Every candidate must bind the portfolio's current evidence CID and an
  exact operator content identity (CID).
* Natural-language implementation bodies and silent IR attachment failures
  are unrepresentable / rejected.
* Runtime model calls remain 0; write authority is never granted.

Predicted symbols: :class:`CandidateFacts`, :class:`CandidatePortfolio`,
:class:`CandidateAdmission`, :func:`build_deterministic_candidate_portfolio`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ..proof.formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    content_identity,
)
from .repair_operator_registry import (
    RepairOperatorRegistry,
    UnknownRepairOperatorError,
    build_default_repair_operator_registry,
    normalize_repair_operator_kind,
)


# ---------------------------------------------------------------------------
# Interfaces / evidence / schemas
# ---------------------------------------------------------------------------

REPAIR_CANDIDATE_INTERFACE: Final[str] = "RepairCandidate@1"
CANDIDATE_ADMISSION_INTERFACE: Final[str] = "CandidateAdmission@1"
CANDIDATE_PORTFOLIO_INTERFACE: Final[str] = "CandidatePortfolio@1"
DCR_CANDIDATE_PORTFOLIO_EVIDENCE: Final[str] = "dcr/candidate-portfolio@1"
DETERMINISTIC_CANDIDATE_PORTFOLIO_VERSION: Final[int] = 1

REPAIR_CANDIDATE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/dcr-repair-candidate@1"
)
CANDIDATE_FACTS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/dcr-candidate-facts@1"
)
CANDIDATE_SCORE_TERMS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/dcr-candidate-score-terms@1"
)
CANDIDATE_PORTFOLIO_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/dcr-candidate-portfolio@1"
)
CANDIDATE_ADMISSION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/dcr-candidate-admission@1"
)
DEFAULT_CANDIDATE_PORTFOLIOS_REL: Final[str] = (
    "data/agent_supervisor/deterministic_contract_repair/candidate-portfolios.json"
)

MAX_CANDIDATES: Final[int] = 32
MAX_ARG_KEYS: Final[int] = 32
MAX_TEXT_BYTES: Final[int] = 4_096
MAX_PATH_BYTES: Final[int] = 1_024
MAX_SCORE: Final[int] = 1_000_000

_PROSE_MARKERS: Final[tuple[str, ...]] = (
    "def ",
    "class ",
    "import ",
    "#!/",
    "function ",
    "private_key",
    "BEGIN ",
    "password=",
    "```",
)

_NL_IMPLEMENTATION_KEYS: Final[frozenset[str]] = frozenset(
    {
        "source_body",
        "implementation",
        "implementation_body",
        "code",
        "patch_body",
        "natural_language",
        "prose",
        "llm_output",
        "model_output",
    }
)


# ---------------------------------------------------------------------------
# Errors / vocabularies
# ---------------------------------------------------------------------------


class CandidatePortfolioError(ContractValidationError):
    """Malformed portfolio input or closed-boundary violation."""


class CandidateAdmissionError(CandidatePortfolioError):
    """Admission inputs violate unique-admission invariants."""


class ApplicabilityStatus(str, Enum):
    """Closed applicability lattice for ranking eligibility."""

    PROVED = "proved"
    UNKNOWN = "unknown"
    REFUTED = "refuted"


class IrAttachmentStatus(str, Enum):
    """IR attachment must be explicit; silent failure is rejected."""

    ATTACHED = "attached"
    NOT_REQUIRED = "not_required"
    FAILED = "failed"
    MISSING = "missing"


class CandidateEligibility(str, Enum):
    ELIGIBLE = "eligible"
    REJECTED = "rejected"
    UNKNOWN = "unknown"


class AdmissionDisposition(str, Enum):
    """Closed portfolio admission outcomes."""

    SELECTED = "selected"
    ABSTAIN = "abstain"
    REJECT = "reject"


class CandidateAdmissionReason(str, Enum):
    """Stable fail-closed reason codes for DCR-062."""

    UNIQUE_WINNER = "unique_winner"
    TIE_ABSTAIN = "tie_abstain"
    UNKNOWN_ABSTAIN = "unknown_abstain"
    NO_ELIGIBLE = "no_eligible_candidate"
    ALL_REFUTED = "all_candidates_refuted"
    STALE_EVIDENCE = "stale_evidence"
    MISSING_OPERATOR_CID = "missing_operator_cid"
    OPERATOR_CID_MISMATCH = "operator_cid_mismatch"
    UNREGISTERED_OPERATOR = "unregistered_operator"
    NATURAL_LANGUAGE_BODY = "natural_language_implementation"
    SILENT_IR_FAILURE = "silent_ir_attachment_failure"
    IR_ATTACHMENT_FAILED = "ir_attachment_failed"
    APPLICABILITY_UNKNOWN = "applicability_unknown"
    APPLICABILITY_REFUTED = "applicability_refuted"
    MISSING_PROOF_RECEIPT = "missing_proof_receipt"
    MISSING_SCORE_TERMS = "missing_score_terms"
    BOUNDS_EXCEEDED = "bounds_exceeded"
    MALFORMED_INPUT = "malformed_input"
    PROPOSAL_ONLY = "proposal_only"
    ZERO_MODEL_CALLS = "zero_model_calls"
    RANKED_BY_SCORE_TERMS = "ranked_by_score_terms"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text(value: Any, name: str, *, required: bool = True, limit: int = MAX_TEXT_BYTES) -> str:
    if value is None:
        if required:
            raise CandidatePortfolioError(f"{name} is required")
        return ""
    if not isinstance(value, str):
        raise CandidatePortfolioError(f"{name} must be a string")
    text = value.strip()
    if required and not text:
        raise CandidatePortfolioError(f"{name} is required")
    if "\x00" in text:
        raise CandidatePortfolioError(f"{name} must not contain NUL")
    if len(text.encode("utf-8")) > limit:
        raise CandidatePortfolioError(
            f"{CandidateAdmissionReason.BOUNDS_EXCEEDED.value}:{name}"
        )
    return text


def _optional_text(value: Any, name: str) -> str:
    return _text(value, name, required=False)


def _nonneg_int(value: Any, name: str, *, maximum: int = MAX_SCORE) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CandidatePortfolioError(f"{name} must be a non-negative integer")
    if value > maximum:
        raise CandidatePortfolioError(
            f"{CandidateAdmissionReason.BOUNDS_EXCEEDED.value}:{name}"
        )
    return value


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise CandidatePortfolioError(f"{name} must be a boolean")
    return value


def _enum(value: Any, enum_cls: type[Enum], name: str) -> Enum:
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(str(value))
    except (TypeError, ValueError) as exc:
        raise CandidatePortfolioError(f"{name} has an unsupported value") from exc


def _ids(
    values: Any,
    name: str,
    *,
    required: bool = False,
    limit: int = MAX_CANDIDATES,
) -> tuple[str, ...]:
    if values is None:
        raw: Sequence[Any] = ()
    elif isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise CandidatePortfolioError(f"{name} must be a sequence of identifiers")
    else:
        raw = values
    if required and not raw:
        raise CandidatePortfolioError(f"{name} is required")
    if len(raw) > limit:
        raise CandidatePortfolioError(
            f"{CandidateAdmissionReason.BOUNDS_EXCEEDED.value}:{name}"
        )
    out: list[str] = []
    for item in raw:
        text = _text(item, name)
        if text not in out:
            out.append(text)
    return tuple(out)


def _path(value: Any, name: str = "path") -> str:
    text = _text(value, name, limit=MAX_PATH_BYTES)
    normalized = text.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts or normalized in {"", "."}:
        raise CandidatePortfolioError(f"{name} must be a bounded relative path")
    return path.as_posix()


def _paths(values: Any, name: str) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise CandidatePortfolioError(f"{name} must be a sequence of paths")
    if len(values) > MAX_ARG_KEYS:
        raise CandidatePortfolioError(
            f"{CandidateAdmissionReason.BOUNDS_EXCEEDED.value}:{name}"
        )
    return tuple(sorted({_path(item, name) for item in values}))


def _assert_body_free(*texts: str) -> None:
    for text in texts:
        lowered = text.lower()
        for marker in _PROSE_MARKERS:
            if marker.lower() in lowered:
                raise CandidatePortfolioError(
                    f"{CandidateAdmissionReason.NATURAL_LANGUAGE_BODY.value}"
                )
        if "\n" in text and any(token in text for token in ("{", "}", ";", "=>")):
            # Multi-line structured source is treated as an implementation body.
            raise CandidatePortfolioError(
                f"{CandidateAdmissionReason.NATURAL_LANGUAGE_BODY.value}"
            )


def _operator_args(value: Any) -> Mapping[str, str]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise CandidatePortfolioError("operator_args must be a mapping")
    if len(value) > MAX_ARG_KEYS:
        raise CandidatePortfolioError(
            f"{CandidateAdmissionReason.BOUNDS_EXCEEDED.value}:operator_args"
        )
    args: dict[str, str] = {}
    for key, item in value.items():
        key_text = _text(str(key), "operator_args key")
        if key_text.casefold().replace("-", "_") in _NL_IMPLEMENTATION_KEYS:
            raise CandidatePortfolioError(
                f"{CandidateAdmissionReason.NATURAL_LANGUAGE_BODY.value}:{key_text}"
            )
        item_text = _text(str(item), f"operator_args[{key_text}]")
        _assert_body_free(item_text)
        args[key_text] = item_text
    return MappingProxyType(dict(sorted(args.items())))


def _operator_cid_for(operator_id: str, kind: str, registry_cid: str) -> str:
    """Derive the exact operator content identity bound into candidates."""

    return content_identity(
        {
            "schema": "ipfs_accelerate_py/agent-supervisor/dcr-operator-binding@1",
            "operator_id": operator_id,
            "kind": kind,
            "registry_cid": registry_cid,
            "interface": "RepairOperator@1",
        }
    )


# ---------------------------------------------------------------------------
# Score terms / facts / candidate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateScoreTerms(CanonicalContract):
    """Fixed-point ranking terms (millionths-scale integers in [0, 1_000_000])."""

    SCHEMA: ClassVar[str] = CANDIDATE_SCORE_TERMS_SCHEMA

    proved_applicability: int
    risk: int
    edit_size: int
    resource_cost: int
    validation_strength: int

    def __post_init__(self) -> None:
        for name in (
            "proved_applicability",
            "risk",
            "edit_size",
            "resource_cost",
            "validation_strength",
        ):
            object.__setattr__(
                self,
                name,
                _nonneg_int(getattr(self, name), name),
            )

    def rank_key(self) -> tuple[int, int, int, int, int]:
        """Lexicographic rank: higher applicability/validation, lower costs."""

        return (
            -self.proved_applicability,
            self.risk,
            self.edit_size,
            self.resource_cost,
            -self.validation_strength,
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "proved_applicability": self.proved_applicability,
            "risk": self.risk,
            "edit_size": self.edit_size,
            "resource_cost": self.resource_cost,
            "validation_strength": self.validation_strength,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CandidateScoreTerms":
        if not isinstance(payload, Mapping):
            raise CandidatePortfolioError("score_terms must be an object")
        return cls(
            proved_applicability=int(payload.get("proved_applicability", -1)),
            risk=int(payload.get("risk", -1)),
            edit_size=int(payload.get("edit_size", -1)),
            resource_cost=int(payload.get("resource_cost", -1)),
            validation_strength=int(payload.get("validation_strength", -1)),
        )


@dataclass(frozen=True)
class CandidateFacts(CanonicalContract):
    """Current-evidence bindings required of every portfolio member."""

    SCHEMA: ClassVar[str] = CANDIDATE_FACTS_SCHEMA

    current_evidence_cid: str
    operator_cid: str
    operator_id: str
    operator_kind: str
    operator_args: Mapping[str, str]
    proof_receipt_cid: str
    write_paths: tuple[str, ...]
    applicability_status: ApplicabilityStatus
    ir_attachment_status: IrAttachmentStatus
    registry_cid: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "current_evidence_cid",
            _text(self.current_evidence_cid, "current_evidence_cid"),
        )
        object.__setattr__(self, "operator_cid", _text(self.operator_cid, "operator_cid"))
        object.__setattr__(self, "operator_id", _text(self.operator_id, "operator_id"))
        object.__setattr__(
            self, "operator_kind", _text(self.operator_kind, "operator_kind")
        )
        object.__setattr__(self, "operator_args", _operator_args(self.operator_args))
        object.__setattr__(
            self,
            "proof_receipt_cid",
            _text(self.proof_receipt_cid, "proof_receipt_cid", required=False),
        )
        object.__setattr__(self, "write_paths", _paths(self.write_paths, "write_paths"))
        object.__setattr__(
            self,
            "applicability_status",
            _enum(self.applicability_status, ApplicabilityStatus, "applicability_status"),
        )
        object.__setattr__(
            self,
            "ir_attachment_status",
            _enum(self.ir_attachment_status, IrAttachmentStatus, "ir_attachment_status"),
        )
        object.__setattr__(
            self, "registry_cid", _optional_text(self.registry_cid, "registry_cid")
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "current_evidence_cid": self.current_evidence_cid,
            "operator_cid": self.operator_cid,
            "operator_id": self.operator_id,
            "operator_kind": self.operator_kind,
            "operator_args": dict(self.operator_args),
            "proof_receipt_cid": self.proof_receipt_cid,
            "write_paths": list(self.write_paths),
            "applicability_status": self.applicability_status.value,
            "ir_attachment_status": self.ir_attachment_status.value,
            "registry_cid": self.registry_cid,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CandidateFacts":
        if not isinstance(payload, Mapping):
            raise CandidatePortfolioError("candidate facts must be an object")
        return cls(
            current_evidence_cid=str(payload.get("current_evidence_cid") or ""),
            operator_cid=str(payload.get("operator_cid") or ""),
            operator_id=str(payload.get("operator_id") or ""),
            operator_kind=str(payload.get("operator_kind") or ""),
            operator_args=payload.get("operator_args") or {},
            proof_receipt_cid=str(payload.get("proof_receipt_cid") or ""),
            write_paths=tuple(payload.get("write_paths") or ()),
            applicability_status=payload.get("applicability_status")
            or ApplicabilityStatus.UNKNOWN,
            ir_attachment_status=payload.get("ir_attachment_status")
            or IrAttachmentStatus.MISSING,
            registry_cid=str(payload.get("registry_cid") or ""),
        )


@dataclass(frozen=True)
class RepairCandidate(CanonicalContract):
    """Body-free ``RepairCandidate@1`` for DCR finite portfolios.

    The candidate carries ranking facts and proof bindings only.  It never
    grants write, transform, or completion authority.
    """

    SCHEMA: ClassVar[str] = REPAIR_CANDIDATE_SCHEMA
    INTERFACE: ClassVar[str] = REPAIR_CANDIDATE_INTERFACE

    facts: CandidateFacts
    score_terms: CandidateScoreTerms
    rejection_reasons: tuple[str, ...] = ()
    eligibility: CandidateEligibility = CandidateEligibility.ELIGIBLE
    grants_write_authority: bool = False
    semantic_authority: bool = False
    runtime_model_calls: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.facts, CandidateFacts):
            if isinstance(self.facts, Mapping):
                object.__setattr__(self, "facts", CandidateFacts.from_dict(self.facts))
            else:
                raise CandidatePortfolioError("facts must be CandidateFacts")
        if not isinstance(self.score_terms, CandidateScoreTerms):
            if isinstance(self.score_terms, Mapping):
                object.__setattr__(
                    self, "score_terms", CandidateScoreTerms.from_dict(self.score_terms)
                )
            else:
                raise CandidatePortfolioError("score_terms must be CandidateScoreTerms")
        object.__setattr__(
            self,
            "rejection_reasons",
            _ids(self.rejection_reasons, "rejection_reasons", limit=MAX_ARG_KEYS),
        )
        object.__setattr__(
            self,
            "eligibility",
            _enum(self.eligibility, CandidateEligibility, "eligibility"),
        )
        # Authority hard-fail closed.
        object.__setattr__(self, "grants_write_authority", False)
        object.__setattr__(self, "semantic_authority", False)
        object.__setattr__(self, "runtime_model_calls", 0)
        if (
            self.eligibility is CandidateEligibility.ELIGIBLE
            and self.rejection_reasons
        ):
            raise CandidatePortfolioError(
                "eligible candidates cannot carry rejection reasons"
            )
        if (
            self.eligibility is not CandidateEligibility.ELIGIBLE
            and not self.rejection_reasons
        ):
            raise CandidatePortfolioError(
                "non-eligible candidates require rejection reasons"
            )

    @property
    def candidate_cid(self) -> str:
        return self.content_id

    @property
    def operator_cid(self) -> str:
        return self.facts.operator_cid

    @property
    def operator_args(self) -> Mapping[str, str]:
        return self.facts.operator_args

    @property
    def proof_receipt_cid(self) -> str:
        return self.facts.proof_receipt_cid

    def evidence_subset(self) -> dict[str, Any]:
        """Project the DCR-062 evidence subset for one candidate."""

        return {
            "candidate_cid": self.candidate_cid,
            "operator_args": dict(self.facts.operator_args),
            "score_terms": self.score_terms.to_dict(),
            "proof_receipt": self.facts.proof_receipt_cid,
            "rejected_reason": list(self.rejection_reasons),
            "operator_cid": self.facts.operator_cid,
            "current_evidence_cid": self.facts.current_evidence_cid,
            "eligibility": self.eligibility.value,
        }

    def _payload(self) -> dict[str, Any]:
        return {
            "interface": REPAIR_CANDIDATE_INTERFACE,
            "facts": self.facts.to_dict(),
            "score_terms": self.score_terms.to_dict(),
            "rejection_reasons": list(self.rejection_reasons),
            "eligibility": self.eligibility.value,
            "grants_write_authority": False,
            "semantic_authority": False,
            "runtime_model_calls": 0,
            "evidence_id": DCR_CANDIDATE_PORTFOLIO_EVIDENCE,
            "version": DETERMINISTIC_CANDIDATE_PORTFOLIO_VERSION,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RepairCandidate":
        if not isinstance(payload, Mapping):
            raise CandidatePortfolioError("repair candidate must be an object")
        return cls(
            facts=payload.get("facts") or {},
            score_terms=payload.get("score_terms") or {},
            rejection_reasons=tuple(payload.get("rejection_reasons") or ()),
            eligibility=payload.get("eligibility") or CandidateEligibility.ELIGIBLE,
        )


# ---------------------------------------------------------------------------
# Portfolio + admission
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidatePortfolio(CanonicalContract):
    """Finite, content-addressed set of ranked repair candidates."""

    SCHEMA: ClassVar[str] = CANDIDATE_PORTFOLIO_SCHEMA
    INTERFACE: ClassVar[str] = CANDIDATE_PORTFOLIO_INTERFACE

    portfolio_id: str
    current_evidence_cid: str
    registry_cid: str
    candidates: tuple[RepairCandidate, ...]
    ranked_candidate_cids: tuple[str, ...] = ()
    runtime_model_calls: int = 0
    grants_write_authority: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "portfolio_id", _text(self.portfolio_id, "portfolio_id")
        )
        object.__setattr__(
            self,
            "current_evidence_cid",
            _text(self.current_evidence_cid, "current_evidence_cid"),
        )
        object.__setattr__(
            self, "registry_cid", _text(self.registry_cid, "registry_cid")
        )
        if not isinstance(self.candidates, Sequence) or isinstance(
            self.candidates, (str, bytes, bytearray)
        ):
            raise CandidatePortfolioError("candidates must be a sequence")
        if len(self.candidates) > MAX_CANDIDATES:
            raise CandidatePortfolioError(
                f"{CandidateAdmissionReason.BOUNDS_EXCEEDED.value}:candidates"
            )
        normalized: list[RepairCandidate] = []
        for item in self.candidates:
            if isinstance(item, RepairCandidate):
                candidate = item
            elif isinstance(item, Mapping):
                candidate = RepairCandidate.from_dict(item)
            else:
                raise CandidatePortfolioError(
                    "candidates must contain RepairCandidate records"
                )
            if candidate.facts.current_evidence_cid != self.current_evidence_cid:
                raise CandidatePortfolioError(
                    f"{CandidateAdmissionReason.STALE_EVIDENCE.value}:"
                    f"{candidate.candidate_cid}"
                )
            if (
                candidate.facts.registry_cid
                and candidate.facts.registry_cid != self.registry_cid
            ):
                raise CandidatePortfolioError(
                    f"{CandidateAdmissionReason.OPERATOR_CID_MISMATCH.value}:"
                    f"registry"
                )
            normalized.append(candidate)
        # Deterministic order by rank key then candidate CID (display order).
        ordered = tuple(
            sorted(
                normalized,
                key=lambda c: (*c.score_terms.rank_key(), c.candidate_cid),
            )
        )
        cids = [item.candidate_cid for item in ordered]
        if len(cids) != len(set(cids)):
            raise CandidatePortfolioError("portfolio contains duplicate candidate CIDs")
        object.__setattr__(self, "candidates", ordered)
        claimed = tuple(self.ranked_candidate_cids or cids)
        if claimed != tuple(cids):
            # Auto-heal empty projection; reject inconsistent caller projection.
            if self.ranked_candidate_cids:
                raise CandidatePortfolioError(
                    "ranked_candidate_cids projection is inconsistent"
                )
            claimed = tuple(cids)
        object.__setattr__(self, "ranked_candidate_cids", claimed)
        object.__setattr__(self, "runtime_model_calls", 0)
        object.__setattr__(self, "grants_write_authority", False)

    @property
    def portfolio_cid(self) -> str:
        return self.content_id

    def eligible(self) -> tuple[RepairCandidate, ...]:
        return tuple(
            item
            for item in self.candidates
            if item.eligibility is CandidateEligibility.ELIGIBLE
        )

    def evidence_subset(self) -> dict[str, Any]:
        return {
            "evidence_id": DCR_CANDIDATE_PORTFOLIO_EVIDENCE,
            "portfolio_id": self.portfolio_id,
            "portfolio_cid": self.portfolio_cid,
            "current_evidence_cid": self.current_evidence_cid,
            "registry_cid": self.registry_cid,
            "candidates": [item.evidence_subset() for item in self.candidates],
            "ranked_candidate_cids": list(self.ranked_candidate_cids),
            "runtime_model_calls": 0,
            "grants_write_authority": False,
        }

    def _payload(self) -> dict[str, Any]:
        return {
            "interface": CANDIDATE_PORTFOLIO_INTERFACE,
            "portfolio_id": self.portfolio_id,
            "current_evidence_cid": self.current_evidence_cid,
            "registry_cid": self.registry_cid,
            "candidates": [item.to_dict() for item in self.candidates],
            "ranked_candidate_cids": list(self.ranked_candidate_cids),
            "runtime_model_calls": 0,
            "grants_write_authority": False,
            "evidence_id": DCR_CANDIDATE_PORTFOLIO_EVIDENCE,
            "version": DETERMINISTIC_CANDIDATE_PORTFOLIO_VERSION,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CandidatePortfolio":
        if not isinstance(payload, Mapping):
            raise CandidatePortfolioError("portfolio must be an object")
        return cls(
            portfolio_id=str(payload.get("portfolio_id") or ""),
            current_evidence_cid=str(payload.get("current_evidence_cid") or ""),
            registry_cid=str(payload.get("registry_cid") or ""),
            candidates=tuple(payload.get("candidates") or ()),
            ranked_candidate_cids=tuple(payload.get("ranked_candidate_cids") or ()),
        )


@dataclass(frozen=True)
class CandidateAdmission(CanonicalContract):
    """``CandidateAdmission@1`` — unique selection or fail-closed abstention."""

    SCHEMA: ClassVar[str] = CANDIDATE_ADMISSION_SCHEMA
    INTERFACE: ClassVar[str] = CANDIDATE_ADMISSION_INTERFACE

    portfolio_cid: str
    disposition: AdmissionDisposition
    selected_candidate_cid: str = ""
    reason_codes: tuple[str, ...] = ()
    rejected: Mapping[str, tuple[str, ...]] = MappingProxyType({})
    ranked_eligible_cids: tuple[str, ...] = ()
    runtime_model_calls: int = 0
    grants_write_authority: bool = False
    proposal_only: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "portfolio_cid", _text(self.portfolio_cid, "portfolio_cid")
        )
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, AdmissionDisposition, "disposition"),
        )
        selected = _optional_text(self.selected_candidate_cid, "selected_candidate_cid")
        object.__setattr__(self, "selected_candidate_cid", selected)
        codes = _ids(self.reason_codes, "reason_codes", limit=MAX_ARG_KEYS)
        if not codes:
            codes = (self.disposition.value,)
        object.__setattr__(self, "reason_codes", codes)
        rejected_raw = self.rejected or {}
        if not isinstance(rejected_raw, Mapping):
            raise CandidateAdmissionError("rejected must be a mapping")
        rejected: dict[str, tuple[str, ...]] = {}
        for key, reasons in rejected_raw.items():
            rejected[_text(str(key), "rejected key")] = _ids(
                reasons, "rejected reasons", limit=MAX_ARG_KEYS
            )
        object.__setattr__(
            self,
            "rejected",
            MappingProxyType(dict(sorted(rejected.items()))),
        )
        object.__setattr__(
            self,
            "ranked_eligible_cids",
            _ids(self.ranked_eligible_cids, "ranked_eligible_cids"),
        )
        object.__setattr__(self, "runtime_model_calls", 0)
        object.__setattr__(self, "grants_write_authority", False)
        object.__setattr__(self, "proposal_only", True)

        if self.disposition is AdmissionDisposition.SELECTED:
            if not selected:
                raise CandidateAdmissionError(
                    "selected disposition requires selected_candidate_cid"
                )
            if (
                CandidateAdmissionReason.UNIQUE_WINNER.value not in self.reason_codes
                and "unique_winner" not in self.reason_codes
            ):
                # Still require unique-winner semantics in reason set.
                raise CandidateAdmissionError(
                    "selected disposition requires unique_winner reason"
                )
        else:
            if selected:
                raise CandidateAdmissionError(
                    "non-selected disposition cannot carry selected_candidate_cid"
                )

    @property
    def admission_cid(self) -> str:
        return self.content_id

    @property
    def ok(self) -> bool:
        return (
            self.disposition is AdmissionDisposition.SELECTED
            and bool(self.selected_candidate_cid)
        )

    def evidence_subset(self) -> dict[str, Any]:
        return {
            "evidence_id": DCR_CANDIDATE_PORTFOLIO_EVIDENCE,
            "admission_cid": self.admission_cid,
            "portfolio_cid": self.portfolio_cid,
            "disposition": self.disposition.value,
            "selected_candidate_cid": self.selected_candidate_cid,
            "reason_codes": list(self.reason_codes),
            "rejected": {key: list(value) for key, value in self.rejected.items()},
            "ranked_eligible_cids": list(self.ranked_eligible_cids),
            "runtime_model_calls": 0,
            "grants_write_authority": False,
            "proposal_only": True,
        }

    def _payload(self) -> dict[str, Any]:
        return {
            "interface": CANDIDATE_ADMISSION_INTERFACE,
            "portfolio_cid": self.portfolio_cid,
            "disposition": self.disposition.value,
            "selected_candidate_cid": self.selected_candidate_cid,
            "reason_codes": list(self.reason_codes),
            "rejected": {key: list(value) for key, value in self.rejected.items()},
            "ranked_eligible_cids": list(self.ranked_eligible_cids),
            "runtime_model_calls": 0,
            "grants_write_authority": False,
            "proposal_only": True,
            "evidence_id": DCR_CANDIDATE_PORTFOLIO_EVIDENCE,
            "version": DETERMINISTIC_CANDIDATE_PORTFOLIO_VERSION,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CandidateAdmission":
        if not isinstance(payload, Mapping):
            raise CandidateAdmissionError("admission must be an object")
        rejected = payload.get("rejected") or {}
        return cls(
            portfolio_cid=str(payload.get("portfolio_cid") or ""),
            disposition=payload.get("disposition") or AdmissionDisposition.ABSTAIN,
            selected_candidate_cid=str(payload.get("selected_candidate_cid") or ""),
            reason_codes=tuple(payload.get("reason_codes") or ()),
            rejected={
                str(key): tuple(value or ())
                for key, value in dict(rejected).items()
            },
            ranked_eligible_cids=tuple(payload.get("ranked_eligible_cids") or ()),
        )


# ---------------------------------------------------------------------------
# Eligibility evaluation
# ---------------------------------------------------------------------------


def evaluate_candidate_eligibility(
    facts: CandidateFacts,
    score_terms: CandidateScoreTerms,
    *,
    current_evidence_cid: str,
    expected_operator_cid: str | None = None,
) -> tuple[CandidateEligibility, tuple[str, ...]]:
    """Classify one candidate against fail-closed admission gates."""

    reasons: list[str] = []

    if facts.current_evidence_cid != current_evidence_cid:
        reasons.append(CandidateAdmissionReason.STALE_EVIDENCE.value)
    if not facts.operator_cid:
        reasons.append(CandidateAdmissionReason.MISSING_OPERATOR_CID.value)
    if expected_operator_cid is not None and facts.operator_cid != expected_operator_cid:
        reasons.append(CandidateAdmissionReason.OPERATOR_CID_MISMATCH.value)
    if not facts.proof_receipt_cid:
        reasons.append(CandidateAdmissionReason.MISSING_PROOF_RECEIPT.value)

    if facts.ir_attachment_status is IrAttachmentStatus.MISSING:
        reasons.append(CandidateAdmissionReason.SILENT_IR_FAILURE.value)
    elif facts.ir_attachment_status is IrAttachmentStatus.FAILED:
        reasons.append(CandidateAdmissionReason.IR_ATTACHMENT_FAILED.value)

    if facts.applicability_status is ApplicabilityStatus.UNKNOWN:
        reasons.append(CandidateAdmissionReason.APPLICABILITY_UNKNOWN.value)
    elif facts.applicability_status is ApplicabilityStatus.REFUTED:
        reasons.append(CandidateAdmissionReason.APPLICABILITY_REFUTED.value)
    elif score_terms.proved_applicability <= 0:
        reasons.append(CandidateAdmissionReason.APPLICABILITY_UNKNOWN.value)

    if not reasons:
        return CandidateEligibility.ELIGIBLE, ()

    # Unknowns abstain (eligibility=unknown); positive refutations reject.
    refuting = {
        CandidateAdmissionReason.APPLICABILITY_REFUTED.value,
        CandidateAdmissionReason.IR_ATTACHMENT_FAILED.value,
        CandidateAdmissionReason.NATURAL_LANGUAGE_BODY.value,
        CandidateAdmissionReason.UNREGISTERED_OPERATOR.value,
    }
    if any(code in refuting for code in reasons):
        return CandidateEligibility.REJECTED, tuple(reasons)
    return CandidateEligibility.UNKNOWN, tuple(reasons)


def admit_candidate_portfolio(portfolio: CandidatePortfolio) -> CandidateAdmission:
    """Admit a uniquely best eligible candidate, else abstain/reject."""

    if not isinstance(portfolio, CandidatePortfolio):
        if isinstance(portfolio, Mapping):
            portfolio = CandidatePortfolio.from_dict(portfolio)
        else:
            raise CandidateAdmissionError("portfolio must be CandidatePortfolio")

    rejected: dict[str, tuple[str, ...]] = {}
    eligible: list[RepairCandidate] = []
    unknown_present = False
    all_refuted = True

    for candidate in portfolio.candidates:
        # Re-evaluate against portfolio evidence for fail-closed replay.
        eligibility, reasons = evaluate_candidate_eligibility(
            candidate.facts,
            candidate.score_terms,
            current_evidence_cid=portfolio.current_evidence_cid,
        )
        # Honor structural rejection reasons already on the candidate.
        if candidate.eligibility is not CandidateEligibility.ELIGIBLE:
            eligibility = candidate.eligibility
            reasons = candidate.rejection_reasons or reasons
        if eligibility is CandidateEligibility.ELIGIBLE:
            all_refuted = False
            eligible.append(candidate)
        else:
            if eligibility is CandidateEligibility.UNKNOWN:
                unknown_present = True
                all_refuted = False
            rejected[candidate.candidate_cid] = reasons

    ranked_eligible = tuple(
        item.candidate_cid
        for item in sorted(
            eligible,
            key=lambda c: (*c.score_terms.rank_key(), c.candidate_cid),
        )
    )

    if not eligible:
        if all_refuted and portfolio.candidates and not unknown_present:
            disposition = AdmissionDisposition.REJECT
            codes = (
                CandidateAdmissionReason.ALL_REFUTED.value,
                CandidateAdmissionReason.NO_ELIGIBLE.value,
                CandidateAdmissionReason.PROPOSAL_ONLY.value,
                CandidateAdmissionReason.ZERO_MODEL_CALLS.value,
            )
        else:
            disposition = AdmissionDisposition.ABSTAIN
            codes = (
                (
                    CandidateAdmissionReason.UNKNOWN_ABSTAIN.value
                    if unknown_present
                    else CandidateAdmissionReason.NO_ELIGIBLE.value
                ),
                CandidateAdmissionReason.PROPOSAL_ONLY.value,
                CandidateAdmissionReason.ZERO_MODEL_CALLS.value,
            )
        return CandidateAdmission(
            portfolio_cid=portfolio.portfolio_cid,
            disposition=disposition,
            reason_codes=codes,
            rejected=rejected,
            ranked_eligible_cids=(),
        )

    # Unique best by score terms only — equal terms are a tie (abstain).
    best = min(eligible, key=lambda c: (*c.score_terms.rank_key(), c.candidate_cid))
    tied = [
        item
        for item in eligible
        if item.score_terms.rank_key() == best.score_terms.rank_key()
    ]
    if len(tied) != 1:
        for item in tied:
            rejected.setdefault(
                item.candidate_cid,
                (CandidateAdmissionReason.TIE_ABSTAIN.value,),
            )
        return CandidateAdmission(
            portfolio_cid=portfolio.portfolio_cid,
            disposition=AdmissionDisposition.ABSTAIN,
            reason_codes=(
                CandidateAdmissionReason.TIE_ABSTAIN.value,
                CandidateAdmissionReason.PROPOSAL_ONLY.value,
                CandidateAdmissionReason.ZERO_MODEL_CALLS.value,
            ),
            rejected=rejected,
            ranked_eligible_cids=ranked_eligible,
        )

    # Non-winners among eligible get explicit not-selected reasons.
    for item in eligible:
        if item.candidate_cid != best.candidate_cid:
            rejected[item.candidate_cid] = (
                CandidateAdmissionReason.RANKED_BY_SCORE_TERMS.value,
            )

    return CandidateAdmission(
        portfolio_cid=portfolio.portfolio_cid,
        disposition=AdmissionDisposition.SELECTED,
        selected_candidate_cid=best.candidate_cid,
        reason_codes=(
            CandidateAdmissionReason.UNIQUE_WINNER.value,
            CandidateAdmissionReason.RANKED_BY_SCORE_TERMS.value,
            CandidateAdmissionReason.PROPOSAL_ONLY.value,
            CandidateAdmissionReason.ZERO_MODEL_CALLS.value,
        ),
        rejected=rejected,
        ranked_eligible_cids=ranked_eligible,
    )


# ---------------------------------------------------------------------------
# Portfolio construction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateNomination:
    """Caller-supplied finite nomination over a registered operator."""

    operator_kind: str
    operator_args: Mapping[str, str] = MappingProxyType({})
    write_paths: tuple[str, ...] = ()
    proof_receipt_cid: str = ""
    applicability_status: ApplicabilityStatus | str = ApplicabilityStatus.PROVED
    ir_attachment_status: IrAttachmentStatus | str = IrAttachmentStatus.ATTACHED
    proved_applicability: int = 800_000
    risk: int = 100_000
    edit_size: int = 1
    resource_cost: int = 100
    validation_strength: int = 700_000
    rejection_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "operator_kind", _text(self.operator_kind, "operator_kind")
        )
        object.__setattr__(self, "operator_args", _operator_args(self.operator_args))
        object.__setattr__(self, "write_paths", _paths(self.write_paths, "write_paths"))
        object.__setattr__(
            self,
            "proof_receipt_cid",
            _optional_text(self.proof_receipt_cid, "proof_receipt_cid"),
        )
        object.__setattr__(
            self,
            "applicability_status",
            _enum(
                self.applicability_status,
                ApplicabilityStatus,
                "applicability_status",
            ),
        )
        object.__setattr__(
            self,
            "ir_attachment_status",
            _enum(
                self.ir_attachment_status,
                IrAttachmentStatus,
                "ir_attachment_status",
            ),
        )
        for name in (
            "proved_applicability",
            "risk",
            "edit_size",
            "resource_cost",
            "validation_strength",
        ):
            object.__setattr__(
                self,
                name,
                _nonneg_int(getattr(self, name), name),
            )
        object.__setattr__(
            self,
            "rejection_reasons",
            _ids(self.rejection_reasons, "rejection_reasons", limit=MAX_ARG_KEYS),
        )


def _coerce_nomination(value: Any) -> CandidateNomination:
    if isinstance(value, CandidateNomination):
        return value
    if not isinstance(value, Mapping):
        raise CandidatePortfolioError("nomination must be a mapping or CandidateNomination")
    return CandidateNomination(
        operator_kind=str(
            value.get("operator_kind")
            or value.get("kind")
            or value.get("operator_id")
            or ""
        ),
        operator_args=value.get("operator_args") or value.get("arguments") or {},
        write_paths=tuple(value.get("write_paths") or ()),
        proof_receipt_cid=str(value.get("proof_receipt_cid") or ""),
        applicability_status=value.get("applicability_status")
        or ApplicabilityStatus.PROVED,
        ir_attachment_status=value.get("ir_attachment_status")
        or IrAttachmentStatus.ATTACHED,
        proved_applicability=int(value.get("proved_applicability", 800_000)),
        risk=int(value.get("risk", 100_000)),
        edit_size=int(value.get("edit_size", 1)),
        resource_cost=int(value.get("resource_cost", 100)),
        validation_strength=int(value.get("validation_strength", 700_000)),
        rejection_reasons=tuple(value.get("rejection_reasons") or ()),
    )


def build_deterministic_candidate_portfolio(
    nominations: Sequence[Any],
    *,
    current_evidence_cid: str,
    portfolio_id: str = "portfolio:dcr062",
    registry: RepairOperatorRegistry | None = None,
    require_registered: bool = True,
) -> CandidatePortfolio:
    """Enumerate registered operators with bounded args into a ranked portfolio.

    Natural-language implementation candidates and silent IR attachment
    failures are rejected into non-eligible members rather than admitted.
    """

    evidence = _text(current_evidence_cid, "current_evidence_cid")
    pid = _text(portfolio_id, "portfolio_id")
    if isinstance(nominations, (str, bytes, bytearray)) or not isinstance(
        nominations, Sequence
    ):
        raise CandidatePortfolioError("nominations must be a sequence")
    if len(nominations) > MAX_CANDIDATES:
        raise CandidatePortfolioError(
            f"{CandidateAdmissionReason.BOUNDS_EXCEEDED.value}:nominations"
        )

    reg = registry if registry is not None else build_default_repair_operator_registry()
    if not isinstance(reg, RepairOperatorRegistry):
        raise CandidatePortfolioError("registry must be RepairOperatorRegistry")
    registry_cid = reg.content_id

    candidates: list[RepairCandidate] = []
    for raw in nominations:
        try:
            nomination = _coerce_nomination(raw)
        except CandidatePortfolioError as exc:
            # Malformed nominations become explicit rejections when possible.
            message = str(exc)
            if CandidateAdmissionReason.NATURAL_LANGUAGE_BODY.value in message:
                # Cannot safely construct facts without args; skip with synthetic reject.
                # Represent via a minimal rejected shell using a placeholder operator.
                kind_text = "add_registration"
                try:
                    if isinstance(raw, Mapping):
                        kind_text = str(
                            raw.get("operator_kind")
                            or raw.get("kind")
                            or "add_registration"
                        )
                except Exception:  # pragma: no cover - defensive
                    kind_text = "add_registration"
                try:
                    kind = normalize_repair_operator_kind(kind_text)
                    operator_id = f"repair-operator:{kind.value}@2"
                except UnknownRepairOperatorError:
                    operator_id = "repair-operator:unknown@0"
                    kind = None
                operator_cid = _operator_cid_for(
                    operator_id,
                    kind.value if kind is not None else "unknown",
                    registry_cid,
                )
                facts = CandidateFacts(
                    current_evidence_cid=evidence,
                    operator_cid=operator_cid,
                    operator_id=operator_id,
                    operator_kind=kind.value if kind is not None else "unknown",
                    operator_args={},
                    proof_receipt_cid="proof:rejected-malformed",
                    write_paths=(),
                    applicability_status=ApplicabilityStatus.REFUTED,
                    ir_attachment_status=IrAttachmentStatus.FAILED,
                    registry_cid=registry_cid,
                )
                score = CandidateScoreTerms(
                    proved_applicability=0,
                    risk=MAX_SCORE,
                    edit_size=MAX_SCORE,
                    resource_cost=MAX_SCORE,
                    validation_strength=0,
                )
                candidates.append(
                    RepairCandidate(
                        facts=facts,
                        score_terms=score,
                        rejection_reasons=(
                            CandidateAdmissionReason.NATURAL_LANGUAGE_BODY.value,
                        ),
                        eligibility=CandidateEligibility.REJECTED,
                    )
                )
                continue
            raise

        try:
            kind = normalize_repair_operator_kind(nomination.operator_kind)
        except UnknownRepairOperatorError:
            if require_registered:
                operator_id = f"repair-operator:{nomination.operator_kind}@0"
                operator_cid = _operator_cid_for(
                    operator_id, nomination.operator_kind, registry_cid
                )
                facts = CandidateFacts(
                    current_evidence_cid=evidence,
                    operator_cid=operator_cid,
                    operator_id=operator_id,
                    operator_kind=nomination.operator_kind,
                    operator_args=nomination.operator_args,
                    proof_receipt_cid=nomination.proof_receipt_cid
                    or "proof:unregistered",
                    write_paths=nomination.write_paths,
                    applicability_status=ApplicabilityStatus.REFUTED,
                    ir_attachment_status=nomination.ir_attachment_status,
                    registry_cid=registry_cid,
                )
                score = CandidateScoreTerms(
                    proved_applicability=0,
                    risk=nomination.risk,
                    edit_size=nomination.edit_size,
                    resource_cost=nomination.resource_cost,
                    validation_strength=0,
                )
                candidates.append(
                    RepairCandidate(
                        facts=facts,
                        score_terms=score,
                        rejection_reasons=(
                            CandidateAdmissionReason.UNREGISTERED_OPERATOR.value,
                        ),
                        eligibility=CandidateEligibility.REJECTED,
                    )
                )
                continue
            raise

        try:
            spec = reg.get(kind)
        except UnknownRepairOperatorError:
            if require_registered:
                operator_id = f"repair-operator:{kind.value}@2"
                operator_cid = _operator_cid_for(operator_id, kind.value, registry_cid)
                facts = CandidateFacts(
                    current_evidence_cid=evidence,
                    operator_cid=operator_cid,
                    operator_id=operator_id,
                    operator_kind=kind.value,
                    operator_args=nomination.operator_args,
                    proof_receipt_cid=nomination.proof_receipt_cid
                    or "proof:unregistered",
                    write_paths=nomination.write_paths,
                    applicability_status=ApplicabilityStatus.REFUTED,
                    ir_attachment_status=nomination.ir_attachment_status,
                    registry_cid=registry_cid,
                )
                score = CandidateScoreTerms(
                    proved_applicability=0,
                    risk=nomination.risk,
                    edit_size=nomination.edit_size,
                    resource_cost=nomination.resource_cost,
                    validation_strength=0,
                )
                candidates.append(
                    RepairCandidate(
                        facts=facts,
                        score_terms=score,
                        rejection_reasons=(
                            CandidateAdmissionReason.UNREGISTERED_OPERATOR.value,
                        ),
                        eligibility=CandidateEligibility.REJECTED,
                    )
                )
                continue
            raise

        operator_id = spec.operator_id
        # Prefer the registry-spec content identity (exact operator CID).
        operator_cid = spec.spec_id
        proof_cid = nomination.proof_receipt_cid or content_identity(
            {
                "role": "proof_receipt",
                "operator_cid": operator_cid,
                "evidence": evidence,
                "kind": kind.value,
            }
        )
        facts = CandidateFacts(
            current_evidence_cid=evidence,
            operator_cid=operator_cid,
            operator_id=operator_id,
            operator_kind=kind.value,
            operator_args=nomination.operator_args,
            proof_receipt_cid=proof_cid,
            write_paths=nomination.write_paths,
            applicability_status=nomination.applicability_status,
            ir_attachment_status=nomination.ir_attachment_status,
            registry_cid=registry_cid,
        )
        score = CandidateScoreTerms(
            proved_applicability=nomination.proved_applicability,
            risk=nomination.risk,
            edit_size=nomination.edit_size,
            resource_cost=nomination.resource_cost,
            validation_strength=nomination.validation_strength,
        )
        eligibility, reasons = evaluate_candidate_eligibility(
            facts,
            score,
            current_evidence_cid=evidence,
            expected_operator_cid=operator_cid,
        )
        if nomination.rejection_reasons:
            eligibility = CandidateEligibility.REJECTED
            reasons = nomination.rejection_reasons
        candidates.append(
            RepairCandidate(
                facts=facts,
                score_terms=score,
                rejection_reasons=reasons,
                eligibility=eligibility,
            )
        )

    return CandidatePortfolio(
        portfolio_id=pid,
        current_evidence_cid=evidence,
        registry_cid=registry_cid,
        candidates=tuple(candidates),
    )


def build_and_admit_candidate_portfolio(
    nominations: Sequence[Any],
    *,
    current_evidence_cid: str,
    portfolio_id: str = "portfolio:dcr062",
    registry: RepairOperatorRegistry | None = None,
) -> tuple[CandidatePortfolio, CandidateAdmission]:
    """Convenience: build the finite portfolio and run unique admission."""

    portfolio = build_deterministic_candidate_portfolio(
        nominations,
        current_evidence_cid=current_evidence_cid,
        portfolio_id=portfolio_id,
        registry=registry,
    )
    return portfolio, admit_candidate_portfolio(portfolio)


def materialize_candidate_portfolios(
    *,
    destination: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Materialize candidate-portfolios.json evidence for DCR-062."""

    evidence = content_identity(
        {"role": "current_evidence", "fixture": "dcr062", "tree": "tree:fixture"}
    )
    portfolio, admission = build_and_admit_candidate_portfolio(
        (
            {
                "operator_kind": "add_registration",
                "operator_args": {
                    "symbol": "echo_tool",
                    "registry_path": (
                        "external/ipfs_accelerate/ipfs_accelerate_py/"
                        "agent_supervisor/planning/fixture_op.py"
                    ),
                },
                "write_paths": (
                    "external/ipfs_accelerate/ipfs_accelerate_py/"
                    "agent_supervisor/planning/fixture_op.py",
                ),
                "proved_applicability": 900_000,
                "risk": 50_000,
                "edit_size": 1,
                "resource_cost": 80,
                "validation_strength": 850_000,
            },
            {
                "operator_kind": "add_import",
                "operator_args": {"module": "fixture_op", "name": "echo_tool"},
                "write_paths": (
                    "external/ipfs_accelerate/ipfs_accelerate_py/"
                    "agent_supervisor/planning/fixture_op.py",
                ),
                "proved_applicability": 700_000,
                "risk": 80_000,
                "edit_size": 2,
                "resource_cost": 120,
                "validation_strength": 600_000,
            },
        ),
        current_evidence_cid=evidence,
        portfolio_id="portfolio:dcr062-fixture",
    )
    payload = {
        "artifact_schema": CANDIDATE_PORTFOLIO_SCHEMA,
        "evidence_id": DCR_CANDIDATE_PORTFOLIO_EVIDENCE,
        "interfaces": {
            "repair_candidate": REPAIR_CANDIDATE_INTERFACE,
            "candidate_admission": CANDIDATE_ADMISSION_INTERFACE,
            "candidate_portfolio": CANDIDATE_PORTFOLIO_INTERFACE,
        },
        "version": DETERMINISTIC_CANDIDATE_PORTFOLIO_VERSION,
        "runtime_model_calls": 0,
        "grants_write_authority": False,
        "portfolio": portfolio.to_dict(),
        "admission": admission.to_dict(),
        "evidence_subset": {
            "portfolio": portfolio.evidence_subset(),
            "admission": admission.evidence_subset(),
        },
    }
    if destination is None:
        root = Path(repo_root) if repo_root else Path.cwd()
        destination = root / DEFAULT_CANDIDATE_PORTFOLIOS_REL
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return payload


__all__ = [
    "AdmissionDisposition",
    "ApplicabilityStatus",
    "CANDIDATE_ADMISSION_INTERFACE",
    "CANDIDATE_ADMISSION_SCHEMA",
    "CANDIDATE_PORTFOLIO_INTERFACE",
    "CANDIDATE_PORTFOLIO_SCHEMA",
    "CandidateAdmission",
    "CandidateAdmissionError",
    "CandidateAdmissionReason",
    "CandidateEligibility",
    "CandidateFacts",
    "CandidateNomination",
    "CandidatePortfolio",
    "CandidatePortfolioError",
    "CandidateScoreTerms",
    "DCR_CANDIDATE_PORTFOLIO_EVIDENCE",
    "DEFAULT_CANDIDATE_PORTFOLIOS_REL",
    "DETERMINISTIC_CANDIDATE_PORTFOLIO_VERSION",
    "IrAttachmentStatus",
    "REPAIR_CANDIDATE_INTERFACE",
    "REPAIR_CANDIDATE_SCHEMA",
    "RepairCandidate",
    "admit_candidate_portfolio",
    "build_and_admit_candidate_portfolio",
    "build_deterministic_candidate_portfolio",
    "evaluate_candidate_eligibility",
    "materialize_candidate_portfolios",
]
