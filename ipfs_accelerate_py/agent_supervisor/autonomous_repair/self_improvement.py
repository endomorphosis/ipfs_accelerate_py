"""DCR-084: bound supervisor self-improvement by evidence and invariants.

Interfaces
----------
* ``BoundedSelfImprovement@1`` — evaluate proposals against sealed baselines.
* ``ImprovementProposal@1`` — body-free proposal for ordering/bounds/operators.

Predicted symbols: :class:`BoundedSelfImprovement`, :class:`ImprovementProposal`.

Normative rules (fail-closed)
-----------------------------
* May tune deterministic ordering/bounds or propose reviewed operators.
* Must not rewrite policy roots, validators, authority, logic semantics, or
  model guards.
* No proposal can lower safety floors or self-admit.
* Unchanged / non-improving proposals converge to no-op with zero new work.
* Runtime model calls remain 0.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar, Final

import hashlib

from ..proof.formal_verification_contracts import ContractValidationError


def _content_id(value: Mapping[str, Any]) -> str:
    """Hash JSON maps that may include metric floats (not proof contracts)."""

    return "sha256:" + hashlib.sha256(
        json.dumps(dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()


BOUNDED_SELF_IMPROVEMENT_INTERFACE: Final[str] = "BoundedSelfImprovement@1"
IMPROVEMENT_PROPOSAL_INTERFACE: Final[str] = "ImprovementProposal@1"
DCR_SELF_IMPROVEMENT_EVIDENCE: Final[str] = "dcr/self-improvement@1"
DCR_SELF_IMPROVEMENT_VERSION: Final[int] = 1
IMPROVEMENT_PROPOSAL_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/dcr-improvement-proposal@1"
)
IMPROVEMENT_EVALUATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/dcr-improvement-evaluation@1"
)
IMPROVEMENT_CATALOG_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/dcr-improvement-proposals-catalog@1"
)
DEFAULT_IMPROVEMENT_PROPOSALS_PATH: Final[str] = (
    "data/agent_supervisor/deterministic_contract_repair/improvement-proposals.json"
)

# Surfaces that self-improvement may never rewrite.
FORBIDDEN_TARGETS: Final[frozenset[str]] = frozenset(
    {
        "policy_root",
        "validator",
        "authority",
        "logic_semantics",
        "model_guard",
        "no_llm_policy",
        "bootstrap_seal",
        "protected_path_policy",
    }
)

# Surfaces that may be proposed with review/shadow evaluation.
ALLOWED_TARGETS: Final[frozenset[str]] = frozenset(
    {
        "ordering_weight",
        "resource_bound",
        "retry_budget",
        "timeout_bound",
        "lane_priority",
        "reviewed_operator",
        "shadow_metric",
    }
)

# Safety floor metric names: candidate must be >= baseline (higher is safer).
SAFETY_FLOOR_METRICS: Final[frozenset[str]] = frozenset(
    {
        "safety_floor",
        "min_proof_strength",
        "min_validation_coverage",
        "model_guard_strictness",
        "authority_strictness",
    }
)


class SelfImprovementError(ContractValidationError):
    """Malformed proposal or closed-boundary violation."""


class ProposalKind(str, Enum):  # noqa: UP042
    ORDERING = "ordering"
    BOUND = "bound"
    REVIEWED_OPERATOR = "reviewed_operator"
    NO_OP = "no_op"


class ProposalDisposition(str, Enum):  # noqa: UP042
    ADMITTED_FOR_REVIEW = "admitted_for_review"
    REJECTED = "rejected"
    NO_OP = "no_op"
    SHADOW_ONLY = "shadow_only"


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SelfImprovementError(f"{name} must be a non-empty string")
    return value.strip()


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SelfImprovementError(f"{name} must be a number")
    return float(value)


@dataclass(frozen=True)
class ImprovementProposal:
    """Body-free self-improvement proposal (``ImprovementProposal@1``)."""

    proposal_id: str
    kind: ProposalKind
    target: str
    parameter: str
    baseline_value: float | str
    candidate_value: float | str
    baseline_metrics: Mapping[str, float] = field(default_factory=dict)
    candidate_metrics: Mapping[str, float] = field(default_factory=dict)
    invariant_ids: tuple[str, ...] = ()
    approval_class: str = "review_required"
    self_admit: bool = False
    rewrites_forbidden_surface: bool = False

    SCHEMA: ClassVar[str] = IMPROVEMENT_PROPOSAL_SCHEMA
    INTERFACE: ClassVar[str] = IMPROVEMENT_PROPOSAL_INTERFACE

    def __post_init__(self) -> None:
        object.__setattr__(self, "proposal_id", _text(self.proposal_id, "proposal_id"))
        if not isinstance(self.kind, ProposalKind):
            raise SelfImprovementError("kind must be ProposalKind")
        object.__setattr__(self, "target", _text(self.target, "target").lower())
        object.__setattr__(self, "parameter", _text(self.parameter, "parameter"))
        object.__setattr__(
            self,
            "baseline_metrics",
            {str(k): float(v) for k, v in dict(self.baseline_metrics).items()},
        )
        object.__setattr__(
            self,
            "candidate_metrics",
            {str(k): float(v) for k, v in dict(self.candidate_metrics).items()},
        )
        object.__setattr__(
            self, "invariant_ids", tuple(str(item) for item in self.invariant_ids)
        )
        object.__setattr__(
            self, "approval_class", _text(self.approval_class, "approval_class")
        )
        object.__setattr__(self, "self_admit", bool(self.self_admit))
        object.__setattr__(
            self, "rewrites_forbidden_surface", bool(self.rewrites_forbidden_surface)
        )
        if self.target in FORBIDDEN_TARGETS:
            object.__setattr__(self, "rewrites_forbidden_surface", True)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "proposal_id": self.proposal_id,
            "kind": self.kind.value,
            "target": self.target,
            "parameter": self.parameter,
            "baseline_value": self.baseline_value,
            "candidate_value": self.candidate_value,
            "baseline_metrics": dict(sorted(self.baseline_metrics.items())),
            "candidate_metrics": dict(sorted(self.candidate_metrics.items())),
            "invariant_ids": list(self.invariant_ids),
            "approval_class": self.approval_class,
            "self_admit": self.self_admit,
            "rewrites_forbidden_surface": self.rewrites_forbidden_surface,
        }
        payload["content_id"] = _content_id(
            {k: v for k, v in payload.items() if k != "content_id"}
        )
        return payload


@dataclass(frozen=True)
class ImprovementEvaluation:
    """Evaluation receipt for one proposal under sealed invariants."""

    proposal_id: str
    disposition: ProposalDisposition
    reason_codes: tuple[str, ...] = ()
    safety_floor_ok: bool = False
    improved: bool = False
    unchanged: bool = False
    grants_self_admission: bool = False
    runtime_model_calls: int = 0
    creates_new_work: bool = False

    SCHEMA: ClassVar[str] = IMPROVEMENT_EVALUATION_SCHEMA
    INTERFACE: ClassVar[str] = BOUNDED_SELF_IMPROVEMENT_INTERFACE

    def __post_init__(self) -> None:
        if not isinstance(self.disposition, ProposalDisposition):
            raise SelfImprovementError("invalid disposition")
        object.__setattr__(
            self, "reason_codes", tuple(str(item) for item in self.reason_codes)
        )
        object.__setattr__(self, "runtime_model_calls", 0)
        object.__setattr__(self, "grants_self_admission", False)

    @property
    def ok(self) -> bool:
        return self.disposition is ProposalDisposition.ADMITTED_FOR_REVIEW

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "proposal_id": self.proposal_id,
            "disposition": self.disposition.value,
            "reason_codes": list(self.reason_codes),
            "safety_floor_ok": self.safety_floor_ok,
            "improved": self.improved,
            "unchanged": self.unchanged,
            "grants_self_admission": False,
            "runtime_model_calls": 0,
            "creates_new_work": self.creates_new_work,
        }
        payload["content_id"] = _content_id(
            {k: v for k, v in payload.items() if k != "content_id"}
        )
        return payload


class BoundedSelfImprovement:
    """Evaluate improvement proposals against sealed baselines and invariants."""

    INTERFACE: ClassVar[str] = BOUNDED_SELF_IMPROVEMENT_INTERFACE

    def __init__(
        self,
        *,
        required_invariants: Sequence[str] = (),
        min_improvement_delta: float = 0.0,
    ) -> None:
        self._required_invariants = tuple(str(item) for item in required_invariants)
        self._min_delta = float(min_improvement_delta)

    def evaluate(self, proposal: ImprovementProposal | Mapping[str, Any]) -> ImprovementEvaluation:
        if not isinstance(proposal, ImprovementProposal):
            proposal = ImprovementProposal(
                proposal_id=str(proposal.get("proposal_id") or "proposal:unknown"),
                kind=ProposalKind(str(proposal.get("kind") or "no_op")),
                target=str(proposal.get("target") or "shadow_metric"),
                parameter=str(proposal.get("parameter") or "value"),
                baseline_value=proposal.get("baseline_value", 0),
                candidate_value=proposal.get("candidate_value", 0),
                baseline_metrics=dict(proposal.get("baseline_metrics") or {}),
                candidate_metrics=dict(proposal.get("candidate_metrics") or {}),
                invariant_ids=tuple(proposal.get("invariant_ids") or ()),
                approval_class=str(proposal.get("approval_class") or "review_required"),
                self_admit=bool(proposal.get("self_admit")),
                rewrites_forbidden_surface=bool(
                    proposal.get("rewrites_forbidden_surface")
                ),
            )

        reasons: list[str] = []

        # Hard reject: forbidden surfaces / self-admission.
        if proposal.rewrites_forbidden_surface or proposal.target in FORBIDDEN_TARGETS:
            return ImprovementEvaluation(
                proposal_id=proposal.proposal_id,
                disposition=ProposalDisposition.REJECTED,
                reason_codes=("forbidden_surface", proposal.target, "no_policy_rewrite"),
                safety_floor_ok=False,
                improved=False,
                unchanged=False,
                creates_new_work=False,
            )
        if proposal.self_admit or proposal.approval_class in {
            "self_admit",
            "auto_admit",
            "self",
        }:
            return ImprovementEvaluation(
                proposal_id=proposal.proposal_id,
                disposition=ProposalDisposition.REJECTED,
                reason_codes=("self_admission_forbidden", "review_required"),
                safety_floor_ok=False,
                improved=False,
                unchanged=False,
                creates_new_work=False,
            )
        if proposal.target not in ALLOWED_TARGETS and proposal.kind is not ProposalKind.NO_OP:
            return ImprovementEvaluation(
                proposal_id=proposal.proposal_id,
                disposition=ProposalDisposition.REJECTED,
                reason_codes=("target_not_allowed", proposal.target),
                safety_floor_ok=False,
                improved=False,
                unchanged=False,
                creates_new_work=False,
            )

        # Required invariants must be cited.
        missing = [
            inv for inv in self._required_invariants if inv not in proposal.invariant_ids
        ]
        if missing:
            return ImprovementEvaluation(
                proposal_id=proposal.proposal_id,
                disposition=ProposalDisposition.REJECTED,
                reason_codes=("missing_invariants", *missing),
                safety_floor_ok=False,
                improved=False,
                unchanged=False,
                creates_new_work=False,
            )

        # Safety floors: candidate must not lower any safety metric.
        safety_ok = True
        for name in SAFETY_FLOOR_METRICS:
            if name not in proposal.baseline_metrics:
                continue
            base = proposal.baseline_metrics[name]
            cand = proposal.candidate_metrics.get(name, base)
            if cand + 1e-12 < base:
                safety_ok = False
                reasons.append(f"safety_floor_lowered:{name}")
        if not safety_ok:
            return ImprovementEvaluation(
                proposal_id=proposal.proposal_id,
                disposition=ProposalDisposition.REJECTED,
                reason_codes=tuple(reasons) + ("safety_floor_violation",),
                safety_floor_ok=False,
                improved=False,
                unchanged=False,
                creates_new_work=False,
            )

        unchanged = proposal.baseline_value == proposal.candidate_value and (
            dict(proposal.baseline_metrics) == dict(proposal.candidate_metrics)
        )
        if unchanged or proposal.kind is ProposalKind.NO_OP:
            return ImprovementEvaluation(
                proposal_id=proposal.proposal_id,
                disposition=ProposalDisposition.NO_OP,
                reason_codes=("unchanged_or_noop", "converge_zero_work"),
                safety_floor_ok=True,
                improved=False,
                unchanged=True,
                creates_new_work=False,
            )

        # Improvement score: prefer higher utility metric when present.
        base_util = float(proposal.baseline_metrics.get("utility", 0.0))
        cand_util = float(proposal.candidate_metrics.get("utility", base_util))
        improved = cand_util > base_util + self._min_delta
        if not improved:
            # Numeric bound/order changes without utility gain are non-improving.
            if isinstance(proposal.baseline_value, (int, float)) and isinstance(
                proposal.candidate_value, (int, float)
            ):
                improved = float(proposal.candidate_value) > float(
                    proposal.baseline_value
                ) + self._min_delta
        if not improved:
            return ImprovementEvaluation(
                proposal_id=proposal.proposal_id,
                disposition=ProposalDisposition.NO_OP,
                reason_codes=("non_improving", "converge_zero_work"),
                safety_floor_ok=True,
                improved=False,
                unchanged=False,
                creates_new_work=False,
            )

        # Improving proposals are admitted only for review / shadow — never
        # auto-applied and never granted provider or policy authority.
        return ImprovementEvaluation(
            proposal_id=proposal.proposal_id,
            disposition=ProposalDisposition.ADMITTED_FOR_REVIEW,
            reason_codes=(
                "improved_under_invariants",
                "safety_floors_held",
                "review_required",
                "no_self_admit",
            ),
            safety_floor_ok=True,
            improved=True,
            unchanged=False,
            creates_new_work=True,
        )

    def evaluate_many(
        self, proposals: Sequence[ImprovementProposal | Mapping[str, Any]]
    ) -> list[ImprovementEvaluation]:
        return [self.evaluate(item) for item in proposals]


def materialize_improvement_proposals(
    *,
    destination: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Materialize improvement-proposals.json evidence for DCR-084."""

    engine = BoundedSelfImprovement(
        required_invariants=("inv:no-llm", "inv:authority-strict"),
        min_improvement_delta=0.0,
    )
    proposals = (
        ImprovementProposal(
            proposal_id="proposal:ordering-utility",
            kind=ProposalKind.ORDERING,
            target="ordering_weight",
            parameter="priority_bias",
            baseline_value=1.0,
            candidate_value=1.2,
            baseline_metrics={"utility": 1.0, "safety_floor": 10.0},
            candidate_metrics={"utility": 1.3, "safety_floor": 10.0},
            invariant_ids=("inv:no-llm", "inv:authority-strict"),
            approval_class="review_required",
        ),
        ImprovementProposal(
            proposal_id="proposal:lower-safety",
            kind=ProposalKind.BOUND,
            target="resource_bound",
            parameter="min_proof_strength",
            baseline_value=5.0,
            candidate_value=3.0,
            baseline_metrics={"utility": 1.0, "min_proof_strength": 5.0},
            candidate_metrics={"utility": 2.0, "min_proof_strength": 3.0},
            invariant_ids=("inv:no-llm", "inv:authority-strict"),
            approval_class="review_required",
        ),
        ImprovementProposal(
            proposal_id="proposal:noop",
            kind=ProposalKind.NO_OP,
            target="shadow_metric",
            parameter="none",
            baseline_value=0,
            candidate_value=0,
            baseline_metrics={"utility": 1.0},
            candidate_metrics={"utility": 1.0},
            invariant_ids=("inv:no-llm", "inv:authority-strict"),
        ),
        ImprovementProposal(
            proposal_id="proposal:forbidden",
            kind=ProposalKind.BOUND,
            target="validator",
            parameter="strictness",
            baseline_value=1,
            candidate_value=0,
            baseline_metrics={"utility": 1.0},
            candidate_metrics={"utility": 9.0},
            invariant_ids=("inv:no-llm", "inv:authority-strict"),
            rewrites_forbidden_surface=True,
        ),
    )
    evaluations = engine.evaluate_many(proposals)
    payload = {
        "schema": IMPROVEMENT_CATALOG_SCHEMA,
        "interface": BOUNDED_SELF_IMPROVEMENT_INTERFACE,
        "proposal_interface": IMPROVEMENT_PROPOSAL_INTERFACE,
        "evidence_id": DCR_SELF_IMPROVEMENT_EVIDENCE,
        "version": DCR_SELF_IMPROVEMENT_VERSION,
        "proposals": [item.to_dict() for item in proposals],
        "evaluations": [item.to_dict() for item in evaluations],
        "runtime_model_calls": 0,
        "grants_self_admission": False,
        "forbidden_targets": sorted(FORBIDDEN_TARGETS),
        "allowed_targets": sorted(ALLOWED_TARGETS),
    }
    base = Path(repo_root).resolve() if repo_root is not None else Path.cwd()
    path = (
        Path(destination)
        if destination is not None
        else base.joinpath(*PurePosixPath(DEFAULT_IMPROVEMENT_PROPOSALS_PATH).parts)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


__all__ = [
    "ALLOWED_TARGETS",
    "BOUNDED_SELF_IMPROVEMENT_INTERFACE",
    "DCR_SELF_IMPROVEMENT_EVIDENCE",
    "DCR_SELF_IMPROVEMENT_VERSION",
    "DEFAULT_IMPROVEMENT_PROPOSALS_PATH",
    "FORBIDDEN_TARGETS",
    "IMPROVEMENT_PROPOSAL_INTERFACE",
    "SAFETY_FLOOR_METRICS",
    "BoundedSelfImprovement",
    "ImprovementEvaluation",
    "ImprovementProposal",
    "ProposalDisposition",
    "ProposalKind",
    "SelfImprovementError",
    "materialize_improvement_proposals",
]
