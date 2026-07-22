"""Schema validation and deterministic scoring for objective plan branches.

Generated alternatives cross a strict boundary before they become scheduler
state.  Ranking uses integer millionths and stable branch identifiers so the
same candidates always produce the same selected and rejected order.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_HALF_UP
from hashlib import sha256
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence


PLAN_EVALUATOR_VERSION = "objective-plan-evaluator-v1"
_BRANCH_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class PlanBranchValidationError(ValueError):
    """Raised when generated plan data does not satisfy the branch schema."""


def _required_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PlanBranchValidationError(f"{field_name} must be a non-empty string")
    if "\x00" in value:
        raise PlanBranchValidationError(f"{field_name} must not contain NUL bytes")
    return value.strip()


def _string_tuple(
    value: Any,
    field_name: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PlanBranchValidationError(f"{field_name} must be an array of strings")
    items: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        normalized = _required_string(item, f"{field_name}[{index}]")
        if normalized not in seen:
            seen.add(normalized)
            items.append(normalized)
    if not items and not allow_empty:
        raise PlanBranchValidationError(f"{field_name} must contain at least one value")
    return tuple(items)


def _repo_paths(value: Any, field_name: str) -> tuple[str, ...]:
    paths = _string_tuple(value, field_name)
    for path in paths:
        candidate = PurePosixPath(path.replace("\\", "/"))
        if candidate.is_absolute() or ".." in candidate.parts:
            raise PlanBranchValidationError(
                f"{field_name} must contain repository-relative paths: {path!r}"
            )
    return paths


def _number(
    value: Any,
    field_name: str,
    *,
    minimum: float,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PlanBranchValidationError(f"{field_name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum or (
        maximum is not None and result > maximum
    ):
        bounds = f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
        raise PlanBranchValidationError(f"{field_name} must be finite and {bounds}")
    return result


def _first(mapping: Mapping[str, Any], name: str, *aliases: str) -> Any:
    for key in (name, *aliases):
        if key in mapping:
            return mapping[key]
    raise PlanBranchValidationError(f"missing required field: {name}")


def _to_millionths(value: int | float | Decimal) -> int:
    return int(
        (Decimal(str(value)) * Decimal(1_000_000)).quantize(
            Decimal("1"), rounding=ROUND_HALF_UP
        )
    )


@dataclass(frozen=True)
class PlanBranch:
    """One schema-validated candidate implementation for a subgoal."""

    branch_id: str
    summary: str
    predicted_files: tuple[str, ...]
    predicted_symbols: tuple[str, ...]
    dependencies: tuple[str, ...]
    validation_commands: tuple[str, ...]
    validation_proof: tuple[str, ...]
    estimated_cost: float
    risk: float
    expected_objective_delta: float
    source: str = "llm_router"

    def __post_init__(self) -> None:
        branch_id = _required_string(self.branch_id, "branch_id")
        if not _BRANCH_ID_RE.fullmatch(branch_id):
            raise PlanBranchValidationError(
                "branch_id must begin with an alphanumeric character and contain only "
                "letters, numbers, '.', '_', ':', or '-'"
            )
        object.__setattr__(self, "branch_id", branch_id)
        object.__setattr__(self, "summary", _required_string(self.summary, "summary"))
        object.__setattr__(
            self,
            "predicted_files",
            _repo_paths(self.predicted_files, "predicted_files"),
        )
        object.__setattr__(
            self,
            "predicted_symbols",
            _string_tuple(self.predicted_symbols, "predicted_symbols"),
        )
        object.__setattr__(
            self,
            "dependencies",
            _string_tuple(self.dependencies, "dependencies", allow_empty=True),
        )
        object.__setattr__(
            self,
            "validation_commands",
            _string_tuple(self.validation_commands, "validation_commands"),
        )
        object.__setattr__(
            self,
            "validation_proof",
            _string_tuple(self.validation_proof, "validation_proof"),
        )
        object.__setattr__(
            self,
            "estimated_cost",
            _number(self.estimated_cost, "estimated_cost", minimum=0.0),
        )
        object.__setattr__(
            self,
            "risk",
            _number(self.risk, "risk", minimum=0.0, maximum=1.0),
        )
        object.__setattr__(
            self,
            "expected_objective_delta",
            _number(
                self.expected_objective_delta,
                "expected_objective_delta",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        object.__setattr__(self, "source", _required_string(self.source, "source"))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanBranch":
        """Validate and construct a branch from a mapping.

        Explicit aliases support trusted deterministic planners. Router JSON is
        checked against the canonical schema before reaching this method.
        """

        if not isinstance(payload, Mapping):
            raise PlanBranchValidationError("each plan branch must be a JSON object")
        allowed = {
            "branch_id",
            "id",
            "summary",
            "plan",
            "predicted_files",
            "files",
            "predicted_symbols",
            "symbols",
            "ast_symbols",
            "dependencies",
            "depends_on",
            "validation_commands",
            "validation",
            "validation_proof",
            "proof",
            "estimated_cost",
            "cost",
            "risk",
            "expected_objective_delta",
            "objective_delta",
            "source",
        }
        unknown = sorted(str(key) for key in payload if key not in allowed)
        if unknown:
            raise PlanBranchValidationError(
                f"unknown plan branch fields: {', '.join(unknown)}"
            )
        return cls(
            branch_id=_first(payload, "branch_id", "id"),
            summary=_first(payload, "summary", "plan"),
            predicted_files=_first(payload, "predicted_files", "files"),
            predicted_symbols=_first(
                payload, "predicted_symbols", "symbols", "ast_symbols"
            ),
            dependencies=_first(payload, "dependencies", "depends_on"),
            validation_commands=_first(
                payload, "validation_commands", "validation"
            ),
            validation_proof=_first(payload, "validation_proof", "proof"),
            estimated_cost=_first(payload, "estimated_cost", "cost"),
            risk=_first(payload, "risk"),
            expected_objective_delta=_first(
                payload, "expected_objective_delta", "objective_delta"
            ),
            source=_first(payload, "source"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible scheduler representation."""

        payload = asdict(self)
        for name in (
            "predicted_files",
            "predicted_symbols",
            "dependencies",
            "validation_commands",
            "validation_proof",
        ):
            payload[name] = list(payload[name])
        return payload

    def validate(self) -> None:
        """Re-run validation for callers handling an already-built branch."""

        PlanBranch.from_dict(self.to_dict())

    def to_profile_g_dict(self) -> dict[str, Any]:
        """Return branch metrics as integer millionths for Profile G."""

        payload = self.to_dict()
        payload.pop("estimated_cost")
        payload.pop("risk")
        payload.pop("expected_objective_delta")
        payload.update(
            {
                "estimated_cost_millionths": _to_millionths(self.estimated_cost),
                "risk_millionths": _to_millionths(self.risk),
                "expected_objective_delta_millionths": _to_millionths(
                    self.expected_objective_delta
                ),
            }
        )
        return payload


PLAN_BRANCH_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Objective plan branches",
    "type": "object",
    "required": ["branches"],
    "additionalProperties": False,
    "properties": {
        "branches": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "branch_id",
                    "summary",
                    "predicted_files",
                    "predicted_symbols",
                    "dependencies",
                    "validation_commands",
                    "validation_proof",
                    "estimated_cost",
                    "risk",
                    "expected_objective_delta",
                    "source",
                ],
                "properties": {
                    "branch_id": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 128,
                    },
                    "summary": {"type": "string", "minLength": 1},
                    "predicted_files": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "string"},
                    },
                    "predicted_symbols": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "string"},
                    },
                    "dependencies": {"type": "array", "items": {"type": "string"}},
                    "validation_commands": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "string"},
                    },
                    "validation_proof": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "string"},
                    },
                    "estimated_cost": {"type": "number", "minimum": 0},
                    "risk": {"type": "number", "minimum": 0, "maximum": 1},
                    "expected_objective_delta": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                    "source": {"const": "llm_router"},
                },
            },
        }
    },
}


ANALYSIS_PROPOSAL_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Low-backlog analysis proposals",
    "type": "object",
    "required": ["proposals"],
    "additionalProperties": False,
    "properties": {
        "proposals": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["branch", "confidence", "novelty", "objective_terms"],
                "additionalProperties": False,
                "properties": {
                    "branch": PLAN_BRANCH_JSON_SCHEMA["properties"]["branches"]["items"],
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "novelty": {"type": "number", "minimum": 0, "maximum": 1},
                    "objective_terms": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "string", "minLength": 1},
                    },
                },
            },
        }
    },
}


@dataclass(frozen=True)
class AnalysisProposal:
    """A plan branch with explicit semantic-analysis evidence."""

    branch: PlanBranch
    confidence: float
    novelty: float
    objective_terms: tuple[str, ...]

    def __post_init__(self) -> None:
        branch = self.branch if isinstance(self.branch, PlanBranch) else PlanBranch.from_dict(self.branch)
        object.__setattr__(self, "branch", branch)
        object.__setattr__(
            self,
            "confidence",
            _number(self.confidence, "confidence", minimum=0.0, maximum=1.0),
        )
        object.__setattr__(
            self,
            "novelty",
            _number(self.novelty, "novelty", minimum=0.0, maximum=1.0),
        )
        object.__setattr__(
            self,
            "objective_terms",
            _string_tuple(self.objective_terms, "objective_terms"),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AnalysisProposal":
        if not isinstance(payload, Mapping):
            raise PlanBranchValidationError("each analysis proposal must be a JSON object")
        allowed = {"branch", "confidence", "novelty", "objective_terms", "proposal_id"}
        required = {"branch", "confidence", "novelty", "objective_terms"}
        unknown = sorted(str(key) for key in payload if key not in allowed)
        missing = sorted(required - set(payload))
        if missing:
            raise PlanBranchValidationError(
                f"analysis proposal is missing required fields: {', '.join(missing)}"
            )
        if unknown:
            raise PlanBranchValidationError(
                f"unknown analysis proposal fields: {', '.join(unknown)}"
            )
        branch = payload["branch"]
        proposal = cls(
            branch=branch if isinstance(branch, PlanBranch) else PlanBranch.from_dict(branch),
            confidence=payload["confidence"],
            novelty=payload["novelty"],
            objective_terms=payload["objective_terms"],
        )
        supplied_id = str(payload.get("proposal_id") or "")
        if supplied_id and supplied_id != proposal.proposal_id:
            raise PlanBranchValidationError("analysis proposal_id does not match canonical content")
        return proposal

    @property
    def proposal_id(self) -> str:
        """Canonical identity independent of an LLM-chosen branch id."""

        material = {
            "summary": " ".join(self.branch.summary.lower().split()),
            "predicted_files": sorted(self.branch.predicted_files),
            "predicted_symbols": sorted(self.branch.predicted_symbols),
            "objective_terms": sorted(term.lower() for term in self.objective_terms),
        }
        digest = sha256(
            json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return f"sha256:{digest}"

    def to_dict(self, *, profile_g: bool = False) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "branch": self.branch.to_profile_g_dict() if profile_g else self.branch.to_dict(),
            "confidence_millionths" if profile_g else "confidence": (
                _to_millionths(self.confidence) if profile_g else self.confidence
            ),
            "novelty_millionths" if profile_g else "novelty": (
                _to_millionths(self.novelty) if profile_g else self.novelty
            ),
            "objective_terms": list(self.objective_terms),
        }


@dataclass(frozen=True)
class RejectedAnalysisProposal:
    proposal: AnalysisProposal
    reason: str

    def to_dict(self, *, profile_g: bool = False) -> dict[str, Any]:
        return {
            "proposal": self.proposal.to_dict(profile_g=profile_g),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class AnalysisProposalEvaluation:
    """Accepted and rejected router candidates after fail-closed policy checks."""

    accepted: tuple[AnalysisProposal, ...]
    rejected: tuple[RejectedAnalysisProposal, ...]
    plan_evaluation: "PlanEvaluation | None" = None

    @property
    def selected(self) -> AnalysisProposal | None:
        if self.plan_evaluation is None:
            return None
        branch_id = self.plan_evaluation.selected.branch_id
        return next(
            (item for item in self.accepted if item.branch.branch_id == branch_id),
            None,
        )

    @property
    def confidence(self) -> float:
        return self.selected.confidence if self.selected is not None else 0.0

    @property
    def novelty(self) -> float:
        return self.selected.novelty if self.selected is not None else 0.0

    def to_dict(self, *, profile_g: bool = False) -> dict[str, Any]:
        return {
            "accepted": [item.to_dict(profile_g=profile_g) for item in self.accepted],
            "rejected": [item.to_dict(profile_g=profile_g) for item in self.rejected],
            "selected": self.selected.to_dict(profile_g=profile_g) if self.selected else None,
            "plan_evaluation": (
                self.plan_evaluation.to_dict(profile_g=profile_g)
                if self.plan_evaluation is not None
                else None
            ),
        }


@dataclass(frozen=True)
class EvaluatedPlanBranch:
    """A branch plus its reproducible score and human-readable rationale."""

    branch: PlanBranch
    score_millionths: int
    rationale: tuple[str, ...]

    def to_dict(self, *, profile_g: bool = False) -> dict[str, Any]:
        return {
            "branch": (
                self.branch.to_profile_g_dict()
                if profile_g
                else self.branch.to_dict()
            ),
            "score_millionths": int(self.score_millionths),
            "rationale": list(self.rationale),
        }


@dataclass(frozen=True)
class PlanEvaluation:
    """Selected branch and all rejected alternatives, in ranked order."""

    selected: PlanBranch
    rejected: tuple[PlanBranch, ...]
    scores: Mapping[str, int]
    rationales: Mapping[str, tuple[str, ...]]
    evaluator_version: str = PLAN_EVALUATOR_VERSION

    def __post_init__(self) -> None:
        branch_ids = [branch.branch_id for branch in (self.selected, *self.rejected)]
        if len(branch_ids) != len(set(branch_ids)):
            raise ValueError("evaluated plan branch ids must be unique")
        scores = {str(key): int(value) for key, value in self.scores.items()}
        rationales = {
            str(key): tuple(str(item) for item in value)
            for key, value in self.rationales.items()
        }
        expected = set(branch_ids)
        if set(scores) != expected or set(rationales) != expected:
            raise ValueError("scores and rationales must cover every evaluated plan branch")
        if any(not rationales[branch_id] for branch_id in branch_ids):
            raise ValueError("every evaluated plan branch must retain a rationale")
        object.__setattr__(self, "scores", MappingProxyType(scores))
        object.__setattr__(self, "rationales", MappingProxyType(rationales))

    @property
    def selected_branch(self) -> PlanBranch:
        return self.selected

    @property
    def ranked(self) -> tuple[PlanBranch, ...]:
        return (self.selected, *self.rejected)

    @property
    def evaluated(self) -> tuple[EvaluatedPlanBranch, ...]:
        return tuple(
            EvaluatedPlanBranch(
                branch=branch,
                score_millionths=int(self.scores[branch.branch_id]),
                rationale=tuple(self.rationales[branch.branch_id]),
            )
            for branch in self.ranked
        )

    def to_dict(self, *, profile_g: bool = False) -> dict[str, Any]:
        encode = PlanBranch.to_profile_g_dict if profile_g else PlanBranch.to_dict
        return {
            "evaluator_version": self.evaluator_version,
            "selected": encode(self.selected),
            "rejected": [encode(item) for item in self.rejected],
            "scores": {key: int(value) for key, value in self.scores.items()},
            "rationales": {
                key: list(value) for key, value in self.rationales.items()
            },
            "selection_rationale": list(
                self.rationales[self.selected.branch_id]
            ),
        }

    def to_profile_g_dict(self) -> dict[str, Any]:
        """Return an integer-only representation safe for Profile G artifacts."""

        return self.to_dict(profile_g=True)


def _score_branch(branch: PlanBranch) -> tuple[int, tuple[str, ...]]:
    delta = Decimal(str(branch.expected_objective_delta))
    risk = Decimal(str(branch.risk))
    cost = Decimal(str(branch.estimated_cost))
    proof_coverage = min(
        Decimal(1),
        Decimal(len(branch.validation_proof))
        / Decimal(max(1, len(branch.validation_commands))),
    )
    cost_efficiency = Decimal(1) / (Decimal(1) + cost)
    score = (
        Decimal("0.55") * delta
        + Decimal("0.25") * (Decimal(1) - risk)
        + Decimal("0.15") * cost_efficiency
        + Decimal("0.05") * proof_coverage
    )
    score_millionths = int(
        (score * Decimal(1_000_000)).quantize(
            Decimal("1"), rounding=ROUND_HALF_UP
        )
    )
    rationale = (
        f"expected objective delta contributes {_to_millionths(Decimal('0.55') * delta)} millionths",
        f"risk adjustment contributes {_to_millionths(Decimal('0.25') * (Decimal(1) - risk))} millionths",
        f"cost efficiency contributes {_to_millionths(Decimal('0.15') * cost_efficiency)} millionths",
        f"validation proof contributes {_to_millionths(Decimal('0.05') * proof_coverage)} millionths",
        f"total deterministic score is {score_millionths} millionths",
    )
    return score_millionths, rationale


def evaluate_plan_branches(
    branches: Iterable[PlanBranch | Mapping[str, Any]],
) -> PlanEvaluation:
    """Rank branches deterministically and retain every rejected alternative."""

    candidates = [
        branch if isinstance(branch, PlanBranch) else PlanBranch.from_dict(branch)
        for branch in branches
    ]
    if not candidates:
        raise ValueError("at least one plan branch is required")
    branch_ids = [branch.branch_id for branch in candidates]
    duplicates = sorted(
        {branch_id for branch_id in branch_ids if branch_ids.count(branch_id) > 1}
    )
    if duplicates:
        raise ValueError(f"plan branch ids must be unique: {', '.join(duplicates)}")

    evaluated = [
        EvaluatedPlanBranch(
            branch=branch,
            score_millionths=score,
            rationale=rationale,
        )
        for branch in candidates
        for score, rationale in [_score_branch(branch)]
    ]
    evaluated.sort(
        key=lambda item: (
            -item.score_millionths,
            item.branch.branch_id,
            json.dumps(item.branch.to_dict(), sort_keys=True, separators=(",", ":")),
        )
    )
    selected = evaluated[0]
    rejected: list[EvaluatedPlanBranch] = []
    for item in evaluated[1:]:
        difference = selected.score_millionths - item.score_millionths
        rejected.append(
            EvaluatedPlanBranch(
                branch=item.branch,
                score_millionths=item.score_millionths,
                rationale=(
                    *item.rationale,
                    f"rejected in favor of {selected.branch.branch_id!r} by "
                    f"{difference} score millionths",
                ),
            )
        )
    ranked = (selected, *rejected)
    return PlanEvaluation(
        selected=selected.branch,
        rejected=tuple(item.branch for item in rejected),
        scores={item.branch.branch_id: item.score_millionths for item in ranked},
        rationales={item.branch.branch_id: item.rationale for item in ranked},
    )


def evaluate_analysis_proposals(
    proposals: Iterable[AnalysisProposal | Mapping[str, Any]],
    *,
    objective_terms: Sequence[str] = (),
    known_proposal_ids: Iterable[str] = (),
    min_confidence: float = 0.65,
    min_novelty: float = 0.35,
    max_novel_proposals: int = 5,
) -> AnalysisProposalEvaluation:
    """Filter and rank semantic proposals under deterministic novelty limits.

    Provider-supplied novelty is only used after canonical identity
    deduplication.  A provider therefore cannot make a repeated proposal novel
    by changing its branch id or numeric claim.
    """

    confidence_floor = _number(
        min_confidence, "min_confidence", minimum=0.0, maximum=1.0
    )
    novelty_floor = _number(min_novelty, "min_novelty", minimum=0.0, maximum=1.0)
    limit = int(max_novel_proposals)
    if limit < 0:
        raise ValueError("max_novel_proposals must be non-negative")
    required_terms = tuple(
        dict.fromkeys(str(item).strip() for item in objective_terms if str(item).strip())
    )
    required_keys = {" ".join(item.lower().split()) for item in required_terms}
    known = {str(item).strip() for item in known_proposal_ids if str(item).strip()}
    candidates = [
        item if isinstance(item, AnalysisProposal) else AnalysisProposal.from_dict(item)
        for item in proposals
    ]
    # Identity first and branch id second makes the decision independent of
    # provider order while retaining stable output for non-identical plans.
    candidates.sort(key=lambda item: (item.proposal_id, item.branch.branch_id))
    accepted: list[AnalysisProposal] = []
    rejected: list[RejectedAnalysisProposal] = []
    observed = set(known)
    accepted_branch_ids: set[str] = set()
    for proposal in candidates:
        proposal_terms = {" ".join(item.lower().split()) for item in proposal.objective_terms}
        if proposal.proposal_id in observed:
            rejected.append(RejectedAnalysisProposal(proposal, "duplicate_candidate"))
            continue
        observed.add(proposal.proposal_id)
        if proposal.branch.branch_id in accepted_branch_ids:
            rejected.append(RejectedAnalysisProposal(proposal, "duplicate_branch_id"))
            continue
        if required_keys and not (required_keys & proposal_terms):
            rejected.append(RejectedAnalysisProposal(proposal, "no_objective_term_coverage"))
            continue
        if proposal.confidence < confidence_floor:
            rejected.append(RejectedAnalysisProposal(proposal, "confidence_below_threshold"))
            continue
        if proposal.novelty < novelty_floor:
            rejected.append(RejectedAnalysisProposal(proposal, "novelty_below_threshold"))
            continue
        if len(accepted) >= limit:
            rejected.append(RejectedAnalysisProposal(proposal, "novelty_limit_reached"))
            continue
        accepted.append(proposal)
        accepted_branch_ids.add(proposal.branch.branch_id)

    accepted.sort(key=lambda item: item.branch.branch_id)
    plan_evaluation = (
        evaluate_plan_branches(item.branch for item in accepted) if accepted else None
    )
    # Preserve deterministic rejection order but make policy failures easy to
    # compare across providers that return candidates in a different order.
    rejected.sort(
        key=lambda item: (
            item.reason,
            item.proposal.proposal_id,
            item.proposal.branch.branch_id,
        )
    )
    return AnalysisProposalEvaluation(
        accepted=tuple(accepted),
        rejected=tuple(rejected),
        plan_evaluation=plan_evaluation,
    )
