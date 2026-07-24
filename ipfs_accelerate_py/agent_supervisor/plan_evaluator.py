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
from enum import Enum
from hashlib import sha256
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence


PLAN_EVALUATOR_VERSION = "objective-plan-evaluator-v1"
PROOF_AWARE_PLAN_EVALUATOR_VERSION = "proof-aware-plan-evaluator-v1"
OBJECTIVE_WORK_EVALUATOR_VERSION = "objective-work-evaluator-v1"
EVIDENCE_AWARE_PLAN_EVALUATOR_VERSION = "evidence-aware-plan-evaluator-v1"
# Objective-heap evidence identity for the non-compensable authority gate.
AUTHORITY_VIOLATION_REJECTION_EVIDENCE_ID = (
    "173075880069453142914839090434430341799"
)
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


# Proof-aware plan evaluation -----------------------------------------------------


def _enum_string(value: Any, field_name: str) -> str:
    """Return the public value of an enum-like plan declaration."""

    return _required_string(getattr(value, "value", value), field_name)


@dataclass(frozen=True)
class ProofAwarePlanCandidate:
    """A plan branch with the proof work needed to make it complete.

    The proof declaration intentionally lives beside, rather than inside,
    :class:`PlanBranch`.  Existing routers and persisted plan artifacts remain
    readable while proof-aware callers cross a strict boundary that requires
    every proof scheduling dimension.  ``candidate_id`` is derived from the
    already validated branch identity; a model cannot provide two identities
    for one plan.
    """

    branch: PlanBranch
    obligation_impact: tuple[str, ...]
    required_assurance: str
    proof_cost: float
    cache_likelihood: float
    dependencies: tuple[str, ...]
    expected_evidence_delta: tuple[str, ...]
    resource_classes: tuple[str, ...]
    proof_critical_path: float = 0.0
    downstream_unlock_value: float = 0.0
    risk: float | None = None
    freshness: float = 1.0

    def __post_init__(self) -> None:
        branch = (
            self.branch
            if isinstance(self.branch, PlanBranch)
            else PlanBranch.from_dict(self.branch)
        )
        object.__setattr__(self, "branch", branch)
        object.__setattr__(
            self,
            "obligation_impact",
            _string_tuple(self.obligation_impact, "obligation_impact"),
        )
        object.__setattr__(
            self,
            "required_assurance",
            _enum_string(self.required_assurance, "required_assurance"),
        )
        object.__setattr__(
            self,
            "proof_cost",
            _number(self.proof_cost, "proof_cost", minimum=0.0),
        )
        object.__setattr__(
            self,
            "cache_likelihood",
            _number(
                self.cache_likelihood,
                "cache_likelihood",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        object.__setattr__(
            self,
            "dependencies",
            _string_tuple(self.dependencies, "dependencies", allow_empty=True),
        )
        object.__setattr__(
            self,
            "expected_evidence_delta",
            _string_tuple(
                self.expected_evidence_delta,
                "expected_evidence_delta",
            ),
        )
        object.__setattr__(
            self,
            "resource_classes",
            _string_tuple(self.resource_classes, "resource_classes"),
        )
        for name in ("proof_critical_path", "downstream_unlock_value"):
            object.__setattr__(
                self,
                name,
                _number(getattr(self, name), name, minimum=0.0),
            )
        candidate_risk = branch.risk if self.risk is None else self.risk
        object.__setattr__(
            self,
            "risk",
            _number(candidate_risk, "risk", minimum=0.0, maximum=1.0),
        )
        object.__setattr__(
            self,
            "freshness",
            _number(self.freshness, "freshness", minimum=0.0, maximum=1.0),
        )

    @property
    def candidate_id(self) -> str:
        return self.branch.branch_id

    @property
    def branch_id(self) -> str:
        """Compatibility identity for scheduler code that handles branches."""

        return self.candidate_id

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProofAwarePlanCandidate":
        if not isinstance(payload, Mapping):
            raise PlanBranchValidationError(
                "each proof-aware plan candidate must be a JSON object"
            )
        allowed = {
            "candidate_id",
            "branch",
            "plan_branch",
            "obligation_impact",
            "obligation_impacts",
            "required_assurance",
            "proof_cost",
            "cache_likelihood",
            "cache_hit_likelihood",
            "dependencies",
            "depends_on",
            "expected_evidence_delta",
            "evidence_delta",
            "resource_classes",
            "required_resource_classes",
            "proof_critical_path",
            "proof_critical_path_length",
            "downstream_unlock_value",
            "risk",
            "freshness",
        }
        unknown = sorted(str(key) for key in payload if key not in allowed)
        if unknown:
            raise PlanBranchValidationError(
                "unknown proof-aware plan candidate fields: "
                + ", ".join(unknown)
            )
        branch_value = _first(payload, "branch", "plan_branch")
        candidate = cls(
            branch=(
                branch_value
                if isinstance(branch_value, PlanBranch)
                else PlanBranch.from_dict(branch_value)
            ),
            obligation_impact=_first(
                payload, "obligation_impact", "obligation_impacts"
            ),
            required_assurance=_first(payload, "required_assurance"),
            proof_cost=_first(payload, "proof_cost"),
            cache_likelihood=_first(
                payload, "cache_likelihood", "cache_hit_likelihood"
            ),
            dependencies=_first(payload, "dependencies", "depends_on"),
            expected_evidence_delta=_first(
                payload, "expected_evidence_delta", "evidence_delta"
            ),
            resource_classes=_first(
                payload, "resource_classes", "required_resource_classes"
            ),
            proof_critical_path=payload.get(
                "proof_critical_path",
                payload.get("proof_critical_path_length", 0.0),
            ),
            downstream_unlock_value=payload.get("downstream_unlock_value", 0.0),
            risk=payload.get("risk"),
            freshness=payload.get("freshness", 1.0),
        )
        supplied_id = str(payload.get("candidate_id") or "").strip()
        if supplied_id and supplied_id != candidate.candidate_id:
            raise PlanBranchValidationError(
                "proof-aware candidate_id must match branch.branch_id"
            )
        return candidate

    def to_dict(self, *, profile_g: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "candidate_id": self.candidate_id,
            "branch": (
                self.branch.to_profile_g_dict()
                if profile_g
                else self.branch.to_dict()
            ),
            "obligation_impact": list(self.obligation_impact),
            "required_assurance": self.required_assurance,
            "dependencies": list(self.dependencies),
            "expected_evidence_delta": list(self.expected_evidence_delta),
            "resource_classes": list(self.resource_classes),
        }
        metrics = {
            "proof_cost": self.proof_cost,
            "cache_likelihood": self.cache_likelihood,
            "proof_critical_path": self.proof_critical_path,
            "downstream_unlock_value": self.downstream_unlock_value,
            "risk": self.risk,
            "freshness": self.freshness,
        }
        if profile_g:
            payload.update(
                {
                    f"{name}_millionths": _to_millionths(value)
                    for name, value in metrics.items()
                }
            )
        else:
            payload.update(metrics)
        return payload

    def to_profile_g_dict(self) -> dict[str, Any]:
        return self.to_dict(profile_g=True)


@dataclass(frozen=True)
class ProofPlanningWeights:
    """Deterministic scoring weights for proof-aware plans."""

    proof_critical_path: float = 0.20
    downstream_unlock_value: float = 0.18
    risk: float = 0.14
    freshness: float = 0.12
    resource_availability: float = 0.12
    cache_likelihood: float = 0.10
    proof_cost: float = 0.07
    evidence_delta: float = 0.05
    dependency_readiness: float = 0.02

    def __post_init__(self) -> None:
        total = Decimal(0)
        for name in self.__dataclass_fields__:
            value = _number(getattr(self, name), name, minimum=0.0)
            object.__setattr__(self, name, value)
            total += Decimal(str(value))
        if total <= 0:
            raise ValueError("at least one proof planning weight must be positive")

    @property
    def total(self) -> Decimal:
        return sum(
            (Decimal(str(getattr(self, name))) for name in self.__dataclass_fields__),
            Decimal(0),
        )

    def to_dict(self, *, profile_g: bool = False) -> dict[str, Any]:
        payload = asdict(self)
        if profile_g:
            return {
                f"{name}_millionths": _to_millionths(value)
                for name, value in payload.items()
            }
        return payload


@dataclass(frozen=True)
class ProofAwarePlanPolicy:
    """Scheduler observations used to rank proof plans.

    Empty availability/dependency observations mean "not reported", not
    "nothing exists".  This keeps evaluation useful before live telemetry is
    attached and avoids turning absence of scheduler context into invented
    negative evidence.
    """

    available_resource_classes: tuple[str, ...] = ()
    satisfied_dependencies: tuple[str, ...] = ()
    weights: ProofPlanningWeights = ProofPlanningWeights()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "available_resource_classes",
            _string_tuple(
                self.available_resource_classes,
                "available_resource_classes",
                allow_empty=True,
            ),
        )
        object.__setattr__(
            self,
            "satisfied_dependencies",
            _string_tuple(
                self.satisfied_dependencies,
                "satisfied_dependencies",
                allow_empty=True,
            ),
        )
        weights = (
            self.weights
            if isinstance(self.weights, ProofPlanningWeights)
            else ProofPlanningWeights(**dict(self.weights))
        )
        object.__setattr__(self, "weights", weights)

    def to_dict(self, *, profile_g: bool = False) -> dict[str, Any]:
        return {
            "available_resource_classes": list(self.available_resource_classes),
            "satisfied_dependencies": list(self.satisfied_dependencies),
            "weights": self.weights.to_dict(profile_g=profile_g),
        }


# A slightly shorter name reads better in generic scheduler configuration.
ProofPlanningPolicy = ProofAwarePlanPolicy


@dataclass(frozen=True)
class EvaluatedProofAwarePlan:
    """One proof-aware alternative with its complete decision trace."""

    candidate: ProofAwarePlanCandidate
    score_millionths: int
    rationale: tuple[str, ...]

    def __post_init__(self) -> None:
        candidate = (
            self.candidate
            if isinstance(self.candidate, ProofAwarePlanCandidate)
            else ProofAwarePlanCandidate.from_dict(self.candidate)
        )
        object.__setattr__(self, "candidate", candidate)
        object.__setattr__(self, "score_millionths", int(self.score_millionths))
        rationale = tuple(str(item).strip() for item in self.rationale if str(item).strip())
        if not rationale:
            raise ValueError("evaluated proof-aware plans require a rationale")
        object.__setattr__(self, "rationale", rationale)

    @property
    def candidate_id(self) -> str:
        return self.candidate.candidate_id

    @property
    def branch(self) -> PlanBranch:
        return self.candidate.branch

    def to_dict(self, *, profile_g: bool = False) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(profile_g=profile_g),
            "score_millionths": self.score_millionths,
            "rationale": list(self.rationale),
        }


@dataclass(frozen=True)
class ProofAwarePlanEvaluation:
    """Selected proof plan and rejected alternatives in deterministic order."""

    selected: EvaluatedProofAwarePlan
    rejected: tuple[EvaluatedProofAwarePlan, ...]
    policy: ProofAwarePlanPolicy
    evaluator_version: str = PROOF_AWARE_PLAN_EVALUATOR_VERSION

    def __post_init__(self) -> None:
        ranked = (self.selected, *self.rejected)
        candidate_ids = [item.candidate_id for item in ranked]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("evaluated proof-aware candidate ids must be unique")

    @property
    def ranked(self) -> tuple[EvaluatedProofAwarePlan, ...]:
        return (self.selected, *self.rejected)

    @property
    def scores(self) -> Mapping[str, int]:
        return MappingProxyType(
            {item.candidate_id: item.score_millionths for item in self.ranked}
        )

    @property
    def rationales(self) -> Mapping[str, tuple[str, ...]]:
        return MappingProxyType(
            {item.candidate_id: item.rationale for item in self.ranked}
        )

    @property
    def selected_candidate(self) -> ProofAwarePlanCandidate:
        return self.selected.candidate

    def to_dict(self, *, profile_g: bool = False) -> dict[str, Any]:
        return {
            "evaluator_version": self.evaluator_version,
            "selected": self.selected.to_dict(profile_g=profile_g),
            "rejected": [
                item.to_dict(profile_g=profile_g) for item in self.rejected
            ],
            "scores": dict(self.scores),
            "rationales": {
                key: list(value) for key, value in self.rationales.items()
            },
            "policy": self.policy.to_dict(profile_g=profile_g),
        }

    def to_profile_g_dict(self) -> dict[str, Any]:
        return self.to_dict(profile_g=True)


def _bounded_benefit(value: float) -> Decimal:
    raw = Decimal(str(value))
    return raw / (Decimal(1) + raw)


def _proof_plan_score(
    candidate: ProofAwarePlanCandidate,
    policy: ProofAwarePlanPolicy,
) -> tuple[int, tuple[str, ...]]:
    weights = policy.weights
    available = {
        item.casefold() for item in policy.available_resource_classes
    }
    required_resources = {
        item.casefold() for item in candidate.resource_classes
    }
    resource_availability = (
        Decimal(1)
        if not available
        else Decimal(len(required_resources & available))
        / Decimal(len(required_resources))
    )
    satisfied = {item.casefold() for item in policy.satisfied_dependencies}
    required_dependencies = {item.casefold() for item in candidate.dependencies}
    dependency_readiness = (
        Decimal(1)
        if not required_dependencies or not policy.satisfied_dependencies
        else Decimal(len(required_dependencies & satisfied))
        / Decimal(len(required_dependencies))
    )
    factors: tuple[tuple[str, Decimal, Decimal], ...] = (
        (
            "proof critical path",
            Decimal(str(weights.proof_critical_path)),
            _bounded_benefit(candidate.proof_critical_path),
        ),
        (
            "downstream unlock value",
            Decimal(str(weights.downstream_unlock_value)),
            _bounded_benefit(candidate.downstream_unlock_value),
        ),
        (
            "risk adjustment",
            Decimal(str(weights.risk)),
            Decimal(1) - Decimal(str(candidate.risk)),
        ),
        (
            "evidence freshness",
            Decimal(str(weights.freshness)),
            Decimal(str(candidate.freshness)),
        ),
        (
            "available resource classes",
            Decimal(str(weights.resource_availability)),
            resource_availability,
        ),
        (
            "cache likelihood",
            Decimal(str(weights.cache_likelihood)),
            Decimal(str(candidate.cache_likelihood)),
        ),
        (
            "proof cost efficiency",
            Decimal(str(weights.proof_cost)),
            Decimal(1) / (Decimal(1) + Decimal(str(candidate.proof_cost))),
        ),
        (
            "expected evidence delta",
            Decimal(str(weights.evidence_delta)),
            min(
                Decimal(1),
                Decimal(len(candidate.expected_evidence_delta)) / Decimal(3),
            ),
        ),
        (
            "dependency readiness",
            Decimal(str(weights.dependency_readiness)),
            dependency_readiness,
        ),
    )
    weighted = tuple(
        (label, weight * factor / weights.total)
        for label, weight, factor in factors
    )
    score = sum((contribution for _, contribution in weighted), Decimal(0))
    score_millionths = _to_millionths(score)
    rationale = tuple(
        f"{label} contributes {_to_millionths(contribution)} millionths"
        for label, contribution in weighted
    ) + (
        f"required assurance is {candidate.required_assurance}",
        f"impacts {len(candidate.obligation_impact)} proof obligation(s)",
        f"total deterministic proof priority is {score_millionths} millionths",
    )
    return score_millionths, rationale


def evaluate_proof_aware_plans(
    candidates: Iterable[ProofAwarePlanCandidate | Mapping[str, Any]],
    *,
    policy: ProofAwarePlanPolicy | None = None,
    available_resource_classes: Sequence[str] | None = None,
    satisfied_dependencies: Sequence[str] | None = None,
    weights: ProofPlanningWeights | Mapping[str, Any] | None = None,
) -> ProofAwarePlanEvaluation:
    """Rank proof-aware candidates without sending proof graphs to a model.

    All scoring inputs are bounded scalar declarations or identifier lists.
    Live resource/dependency observations may be supplied directly for
    convenience, but cannot be mixed ambiguously with a policy object.
    """

    if policy is not None and any(
        value is not None
        for value in (
            available_resource_classes,
            satisfied_dependencies,
            weights,
        )
    ):
        raise ValueError(
            "policy cannot be combined with direct proof planning overrides"
        )
    if policy is None:
        resolved_weights = (
            ProofPlanningWeights()
            if weights is None
            else (
                weights
                if isinstance(weights, ProofPlanningWeights)
                else ProofPlanningWeights(**dict(weights))
            )
        )
        policy = ProofAwarePlanPolicy(
            available_resource_classes=tuple(available_resource_classes or ()),
            satisfied_dependencies=tuple(satisfied_dependencies or ()),
            weights=resolved_weights,
        )
    elif not isinstance(policy, ProofAwarePlanPolicy):
        policy = ProofAwarePlanPolicy(**dict(policy))

    normalized = [
        item
        if isinstance(item, ProofAwarePlanCandidate)
        else ProofAwarePlanCandidate.from_dict(item)
        for item in candidates
    ]
    if not normalized:
        raise ValueError("at least one proof-aware plan candidate is required")
    candidate_ids = [item.candidate_id for item in normalized]
    duplicates = sorted(
        {
            candidate_id
            for candidate_id in candidate_ids
            if candidate_ids.count(candidate_id) > 1
        }
    )
    if duplicates:
        raise ValueError(
            "proof-aware candidate ids must be unique: " + ", ".join(duplicates)
        )

    evaluated = [
        EvaluatedProofAwarePlan(candidate, score, rationale)
        for candidate in normalized
        for score, rationale in [_proof_plan_score(candidate, policy)]
    ]
    evaluated.sort(
        key=lambda item: (
            -item.score_millionths,
            item.candidate_id,
            json.dumps(
                item.candidate.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
    )
    winner = evaluated[0]
    rejected = tuple(
        EvaluatedProofAwarePlan(
            candidate=item.candidate,
            score_millionths=item.score_millionths,
            rationale=(
                *item.rationale,
                f"rejected in favor of {winner.candidate_id!r} by "
                f"{winner.score_millionths - item.score_millionths} "
                "priority millionths",
            ),
        )
        for item in evaluated[1:]
    )
    return ProofAwarePlanEvaluation(
        selected=winner,
        rejected=rejected,
        policy=policy,
    )


evaluate_proof_aware_plan_candidates = evaluate_proof_aware_plans


# Evidence-aware plan evaluation ------------------------------------------------


def _boolean(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise PlanBranchValidationError(f"{field_name} must be boolean")
    return value


def _non_negative_integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PlanBranchValidationError(
            f"{field_name} must be a non-negative integer"
        )
    return value


def _casefold_set(values: Iterable[str]) -> set[str]:
    return {" ".join(value.casefold().split()) for value in values}


def _missing(required: Iterable[str], observed: Iterable[str]) -> tuple[str, ...]:
    observed_keys = _casefold_set(observed)
    return tuple(
        item for item in required
        if " ".join(item.casefold().split()) not in observed_keys
    )


@dataclass(frozen=True)
class EvidenceAwarePlanCandidate:
    """A complete, externally assessable implementation-plan declaration.

    Candidate generation and evaluation deliberately remain separate.  Values
    such as ``validated_assumptions`` and ``supported_semantics`` are bindings
    to evidence the caller has already inspected; evaluation still intersects
    them with the frozen policy's trusted sets rather than treating a model's
    self-report as authority.
    """

    branch: PlanBranch
    covered_acceptance_criteria: tuple[str, ...]
    covered_evidence_terms: tuple[str, ...]
    assumptions: tuple[str, ...]
    validated_assumptions: tuple[str, ...]
    semantic_requirements: tuple[str, ...]
    supported_semantics: tuple[str, ...]
    dependencies: tuple[str, ...]
    critical_path: tuple[str, ...]
    unresolved_conflicts: tuple[str, ...]
    changed_scopes: tuple[str, ...]
    authorized_scopes: tuple[str, ...]
    authority_violations: tuple[str, ...]
    validation_feasible: bool
    proof_feasible: bool
    novelty: float
    resource_classes: tuple[str, ...]
    estimated_resource_cost: float
    estimated_tokens: int

    def __post_init__(self) -> None:
        branch = (
            self.branch
            if isinstance(self.branch, PlanBranch)
            else PlanBranch.from_dict(self.branch)
        )
        object.__setattr__(self, "branch", branch)
        required_string_fields = (
            "covered_acceptance_criteria",
            "covered_evidence_terms",
            "changed_scopes",
        )
        optional_string_fields = (
            "assumptions",
            "validated_assumptions",
            "semantic_requirements",
            "supported_semantics",
            "dependencies",
            "critical_path",
            "unresolved_conflicts",
            "authorized_scopes",
            "authority_violations",
            "resource_classes",
        )
        for name in required_string_fields:
            object.__setattr__(
                self, name, _string_tuple(getattr(self, name), name)
            )
        for name in optional_string_fields:
            object.__setattr__(
                self,
                name,
                _string_tuple(getattr(self, name), name, allow_empty=True),
            )
        if _missing(self.validated_assumptions, self.assumptions):
            raise PlanBranchValidationError(
                "validated_assumptions must be a subset of assumptions"
            )
        if _missing(self.supported_semantics, self.semantic_requirements):
            raise PlanBranchValidationError(
                "supported_semantics must be a subset of semantic_requirements"
            )
        if _missing(self.critical_path, self.dependencies):
            raise PlanBranchValidationError(
                "critical_path must be a subset of dependencies"
            )
        for name in ("validation_feasible", "proof_feasible"):
            object.__setattr__(
                self, name, _boolean(getattr(self, name), name)
            )
        object.__setattr__(
            self,
            "novelty",
            _number(self.novelty, "novelty", minimum=0.0, maximum=1.0),
        )
        object.__setattr__(
            self,
            "estimated_resource_cost",
            _number(
                self.estimated_resource_cost,
                "estimated_resource_cost",
                minimum=0.0,
            ),
        )
        object.__setattr__(
            self,
            "estimated_tokens",
            _non_negative_integer(self.estimated_tokens, "estimated_tokens"),
        )

    @property
    def candidate_id(self) -> str:
        return self.branch.branch_id

    @property
    def branch_id(self) -> str:
        return self.candidate_id

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvidenceAwarePlanCandidate":
        if not isinstance(payload, Mapping):
            raise PlanBranchValidationError(
                "each evidence-aware plan candidate must be a JSON object"
            )
        fields = set(cls.__dataclass_fields__)
        allowed = fields | {"candidate_id", "plan_branch"}
        unknown = sorted(str(key) for key in payload if key not in allowed)
        if unknown:
            raise PlanBranchValidationError(
                "unknown evidence-aware plan candidate fields: "
                + ", ".join(unknown)
            )
        required = fields - {"branch"}
        missing = sorted(name for name in required if name not in payload)
        if "branch" not in payload and "plan_branch" not in payload:
            missing.append("branch")
        if missing:
            raise PlanBranchValidationError(
                "evidence-aware plan candidate is missing required fields: "
                + ", ".join(sorted(set(missing)))
            )
        values = {
            name: payload[name]
            for name in required
        }
        branch_value = payload.get("branch", payload.get("plan_branch"))
        candidate = cls(
            branch=(
                branch_value
                if isinstance(branch_value, PlanBranch)
                else PlanBranch.from_dict(branch_value)
            ),
            **values,
        )
        supplied_id = str(payload.get("candidate_id") or "").strip()
        if supplied_id and supplied_id != candidate.candidate_id:
            raise PlanBranchValidationError(
                "evidence-aware candidate_id must match branch.branch_id"
            )
        return candidate

    def to_dict(self, *, profile_g: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "candidate_id": self.candidate_id,
            "branch": (
                self.branch.to_profile_g_dict()
                if profile_g
                else self.branch.to_dict()
            ),
        }
        for name in (
            "covered_acceptance_criteria",
            "covered_evidence_terms",
            "assumptions",
            "validated_assumptions",
            "semantic_requirements",
            "supported_semantics",
            "dependencies",
            "critical_path",
            "unresolved_conflicts",
            "changed_scopes",
            "authorized_scopes",
            "authority_violations",
            "resource_classes",
        ):
            payload[name] = list(getattr(self, name))
        payload.update(
            {
                "validation_feasible": self.validation_feasible,
                "proof_feasible": self.proof_feasible,
            }
        )
        if profile_g:
            payload.update(
                {
                    "novelty_millionths": _to_millionths(self.novelty),
                    "estimated_resource_cost_millionths": _to_millionths(
                        self.estimated_resource_cost
                    ),
                    "estimated_tokens": self.estimated_tokens,
                }
            )
        else:
            payload.update(
                {
                    "novelty": self.novelty,
                    "estimated_resource_cost": self.estimated_resource_cost,
                    "estimated_tokens": self.estimated_tokens,
                }
            )
        return payload


@dataclass(frozen=True)
class EvidenceAwarePlanPolicy:
    """Frozen goal, trusted observations, and finite feasibility bounds."""

    acceptance_criteria: tuple[str, ...]
    evidence_terms: tuple[str, ...]
    trusted_assumptions: tuple[str, ...] = ()
    supported_semantics: tuple[str, ...] = ()
    satisfied_dependencies: tuple[str, ...] = ()
    allowed_scopes: tuple[str, ...] = ()
    available_resource_classes: tuple[str, ...] = ()
    max_estimated_resource_cost: float = 1_000_000.0
    max_estimated_tokens: int = 1_000_000_000
    min_novelty: float = 0.0
    require_validation: bool = True
    require_proof: bool = True

    def __post_init__(self) -> None:
        for name in ("acceptance_criteria", "evidence_terms"):
            object.__setattr__(
                self, name, _string_tuple(getattr(self, name), name)
            )
        for name in (
            "trusted_assumptions",
            "supported_semantics",
            "satisfied_dependencies",
            "allowed_scopes",
            "available_resource_classes",
        ):
            object.__setattr__(
                self,
                name,
                _string_tuple(getattr(self, name), name, allow_empty=True),
            )
        object.__setattr__(
            self,
            "max_estimated_resource_cost",
            _number(
                self.max_estimated_resource_cost,
                "max_estimated_resource_cost",
                minimum=0.0,
            ),
        )
        object.__setattr__(
            self,
            "max_estimated_tokens",
            _non_negative_integer(
                self.max_estimated_tokens, "max_estimated_tokens"
            ),
        )
        object.__setattr__(
            self,
            "min_novelty",
            _number(
                self.min_novelty, "min_novelty", minimum=0.0, maximum=1.0
            ),
        )
        for name in ("require_validation", "require_proof"):
            object.__setattr__(
                self, name, _boolean(getattr(self, name), name)
            )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvidenceAwarePlanPolicy":
        if not isinstance(payload, Mapping):
            raise PlanBranchValidationError(
                "evidence-aware plan policy must be a JSON object"
            )
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(str(key) for key in payload if key not in allowed)
        if unknown:
            raise PlanBranchValidationError(
                "unknown evidence-aware plan policy fields: "
                + ", ".join(unknown)
            )
        missing = [
            name
            for name in ("acceptance_criteria", "evidence_terms")
            if name not in payload
        ]
        if missing:
            raise PlanBranchValidationError(
                "evidence-aware plan policy is missing required fields: "
                + ", ".join(missing)
            )
        return cls(**dict(payload))

    def to_dict(self, *, profile_g: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            name: list(getattr(self, name))
            for name in (
                "acceptance_criteria",
                "evidence_terms",
                "trusted_assumptions",
                "supported_semantics",
                "satisfied_dependencies",
                "allowed_scopes",
                "available_resource_classes",
            )
        }
        payload.update(
            {
                "max_estimated_tokens": self.max_estimated_tokens,
                "min_novelty_millionths" if profile_g else "min_novelty": (
                    _to_millionths(self.min_novelty)
                    if profile_g
                    else self.min_novelty
                ),
                (
                    "max_estimated_resource_cost_millionths"
                    if profile_g
                    else "max_estimated_resource_cost"
                ): (
                    _to_millionths(self.max_estimated_resource_cost)
                    if profile_g
                    else self.max_estimated_resource_cost
                ),
                "require_validation": self.require_validation,
                "require_proof": self.require_proof,
            }
        )
        return payload


class PlanEvaluationDimension(str, Enum):
    ACCEPTANCE_AND_EVIDENCE = "acceptance_and_evidence"
    ASSUMPTIONS_AND_SEMANTICS = "assumptions_and_semantics"
    DEPENDENCIES_AND_CRITICAL_PATH = "dependencies_and_critical_path"
    CONFLICT_SCOPE_AND_AUTHORITY = "conflict_scope_and_authority"
    VALIDATION_AND_PROOF = "validation_and_proof"
    NOVELTY = "novelty"
    RESOURCES_AND_TOKEN_COST = "resources_and_token_cost"


@dataclass(frozen=True)
class PlanDimensionAssessment:
    dimension: PlanEvaluationDimension
    passed: bool
    hard_gate: bool
    score_millionths: int
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "dimension", PlanEvaluationDimension(self.dimension)
        )
        for name in ("passed", "hard_gate"):
            object.__setattr__(
                self, name, _boolean(getattr(self, name), name)
            )
        score = _non_negative_integer(
            self.score_millionths, "score_millionths"
        )
        if score > 1_000_000:
            raise PlanBranchValidationError(
                "score_millionths must not exceed 1000000"
            )
        object.__setattr__(self, "score_millionths", score)
        reasons = tuple(
            str(item).strip() for item in self.reasons if str(item).strip()
        )
        if not reasons:
            raise PlanBranchValidationError(
                "dimension assessment requires at least one reason"
            )
        object.__setattr__(self, "reasons", reasons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension.value,
            "passed": self.passed,
            "hard_gate": self.hard_gate,
            "score_millionths": self.score_millionths,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class EvaluatedEvidenceAwarePlan:
    candidate: EvidenceAwarePlanCandidate
    score_millionths: int
    dimensions: tuple[PlanDimensionAssessment, ...]
    hard_gate_failures: tuple[str, ...]

    def __post_init__(self) -> None:
        candidate = (
            self.candidate
            if isinstance(self.candidate, EvidenceAwarePlanCandidate)
            else EvidenceAwarePlanCandidate.from_dict(self.candidate)
        )
        object.__setattr__(self, "candidate", candidate)
        object.__setattr__(
            self,
            "score_millionths",
            _non_negative_integer(self.score_millionths, "score_millionths"),
        )
        dimensions = tuple(self.dimensions)
        expected = set(PlanEvaluationDimension)
        actual = {item.dimension for item in dimensions}
        if actual != expected or len(dimensions) != len(expected):
            raise PlanBranchValidationError(
                "evaluation must contain every plan dimension exactly once"
            )
        object.__setattr__(self, "dimensions", dimensions)
        failures = tuple(
            str(item).strip()
            for item in self.hard_gate_failures
            if str(item).strip()
        )
        object.__setattr__(self, "hard_gate_failures", failures)
        expected_failures = {
            item.dimension.value
            for item in dimensions
            if item.hard_gate and not item.passed
        }
        if set(failures) != expected_failures:
            raise PlanBranchValidationError(
                "hard_gate_failures must match failed hard-gate dimensions"
            )

    @property
    def candidate_id(self) -> str:
        return self.candidate.candidate_id

    @property
    def admissible(self) -> bool:
        return not self.hard_gate_failures

    @property
    def rationale(self) -> tuple[str, ...]:
        return tuple(
            reason
            for assessment in self.dimensions
            for reason in assessment.reasons
        )

    def to_dict(self, *, profile_g: bool = False) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(profile_g=profile_g),
            "score_millionths": self.score_millionths,
            "admissible": self.admissible,
            "hard_gate_failures": list(self.hard_gate_failures),
            "dimensions": [item.to_dict() for item in self.dimensions],
            "rationale": list(self.rationale),
        }


@dataclass(frozen=True)
class EvidenceAwarePlanEvaluation:
    selected: EvaluatedEvidenceAwarePlan | None
    admissible: tuple[EvaluatedEvidenceAwarePlan, ...]
    rejected: tuple[EvaluatedEvidenceAwarePlan, ...]
    policy: EvidenceAwarePlanPolicy
    evaluator_version: str = EVIDENCE_AWARE_PLAN_EVALUATOR_VERSION

    def __post_init__(self) -> None:
        if self.evaluator_version != EVIDENCE_AWARE_PLAN_EVALUATOR_VERSION:
            raise PlanBranchValidationError(
                "unsupported evidence-aware plan evaluator version"
            )
        all_items = (*self.admissible, *self.rejected)
        ids = [item.candidate_id for item in all_items]
        if len(ids) != len(set(ids)):
            raise PlanBranchValidationError(
                "evaluated evidence-aware candidate ids must be unique"
            )
        if any(not item.admissible for item in self.admissible):
            raise PlanBranchValidationError(
                "admissible plans must pass every hard gate"
            )
        if any(item.admissible for item in self.rejected):
            raise PlanBranchValidationError(
                "rejected plans must fail at least one hard gate"
            )
        if self.selected is not None and (
            not self.selected.admissible
            or not self.admissible
            or self.selected != self.admissible[0]
        ):
            raise PlanBranchValidationError(
                "selected plan must be the first admissible plan"
            )
        if self.selected is None and self.admissible:
            raise PlanBranchValidationError(
                "an admissible plan must be selected"
            )

    @property
    def ranked(self) -> tuple[EvaluatedEvidenceAwarePlan, ...]:
        return (*self.admissible, *self.rejected)

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        """Return no authority evidence without trusted gate provenance.

        Candidate declarations are sufficient for deterministic rejection and
        diagnostics, but they are not authoritative observations.  The
        adaptive-planner boundary combines this evaluation with trusted hard
        constraint receipts and emits the content-addressed objective witness.
        ``requirement_ids`` in :meth:`to_dict` remains routing metadata.
        """

        return ()

    def to_dict(self, *, profile_g: bool = False) -> dict[str, Any]:
        return {
            "evaluator_version": self.evaluator_version,
            "requirement_ids": [
                AUTHORITY_VIOLATION_REJECTION_EVIDENCE_ID
            ],
            "evidence_ids": list(self.evidence_ids),
            "selected": (
                self.selected.to_dict(profile_g=profile_g)
                if self.selected is not None
                else None
            ),
            "admissible": [
                item.to_dict(profile_g=profile_g) for item in self.admissible
            ],
            "rejected": [
                item.to_dict(profile_g=profile_g) for item in self.rejected
            ],
            "policy": self.policy.to_dict(profile_g=profile_g),
        }

    def to_profile_g_dict(self) -> dict[str, Any]:
        return self.to_dict(profile_g=True)


def _dimension(
    dimension: PlanEvaluationDimension,
    *,
    failures: Sequence[str],
    successes: Sequence[str],
    hard_gate: bool,
    score: Decimal,
) -> PlanDimensionAssessment:
    passed = not failures
    reasons = tuple(failures if failures else successes)
    return PlanDimensionAssessment(
        dimension=dimension,
        passed=passed,
        hard_gate=hard_gate,
        score_millionths=_to_millionths(max(Decimal(0), min(Decimal(1), score))),
        reasons=reasons,
    )


def _assess_evidence_aware_plan(
    candidate: EvidenceAwarePlanCandidate,
    policy: EvidenceAwarePlanPolicy,
) -> EvaluatedEvidenceAwarePlan:
    missing_acceptance = _missing(
        policy.acceptance_criteria, candidate.covered_acceptance_criteria
    )
    missing_evidence = _missing(
        policy.evidence_terms, candidate.covered_evidence_terms
    )
    coverage_failures = tuple(
        f"missing frozen acceptance criterion: {item}"
        for item in missing_acceptance
    ) + tuple(
        f"missing objective evidence term: {item}" for item in missing_evidence
    )
    covered_count = (
        len(policy.acceptance_criteria)
        + len(policy.evidence_terms)
        - len(missing_acceptance)
        - len(missing_evidence)
    )
    coverage_total = len(policy.acceptance_criteria) + len(policy.evidence_terms)
    coverage_score = Decimal(covered_count) / Decimal(coverage_total)

    unvalidated = _missing(candidate.assumptions, candidate.validated_assumptions)
    untrusted = _missing(candidate.assumptions, policy.trusted_assumptions)
    locally_unsupported = _missing(
        candidate.semantic_requirements, candidate.supported_semantics
    )
    policy_unsupported = _missing(
        candidate.semantic_requirements, policy.supported_semantics
    )
    assumption_failures = tuple(
        f"assumption lacks validation evidence: {item}" for item in unvalidated
    ) + tuple(
        f"assumption is not trusted by frozen policy: {item}" for item in untrusted
    ) + tuple(
        f"required semantics are not supported by candidate evidence: {item}"
        for item in locally_unsupported
    ) + tuple(
        f"required semantics are unsupported by frozen policy: {item}"
        for item in policy_unsupported
    )

    unsatisfied = _missing(
        candidate.dependencies, policy.satisfied_dependencies
    )
    critical_unsatisfied = _missing(
        candidate.critical_path, policy.satisfied_dependencies
    )
    dependency_failures = tuple(
        f"unsatisfied dependency: {item}" for item in unsatisfied
    )
    dependency_successes = (
        f"{len(candidate.dependencies)} dependencies are satisfied",
        f"{len(candidate.critical_path)} critical-path dependencies are ready",
    )

    unauthorized_candidate = _missing(
        candidate.changed_scopes, candidate.authorized_scopes
    )
    unauthorized_policy = _missing(
        candidate.changed_scopes, policy.allowed_scopes
    )
    conflict_failures = tuple(
        f"unresolved conflict: {item}" for item in candidate.unresolved_conflicts
    ) + tuple(
        f"candidate authority does not cover changed scope: {item}"
        for item in unauthorized_candidate
    ) + tuple(
        f"frozen policy does not authorize changed scope: {item}"
        for item in unauthorized_policy
    ) + tuple(
        f"authority violation: {item}" for item in candidate.authority_violations
    )

    validation_failures: tuple[str, ...] = ()
    if policy.require_validation and not candidate.validation_feasible:
        validation_failures += ("validation commands are not feasible",)
    if policy.require_proof and not candidate.proof_feasible:
        validation_failures += ("required proof is not feasible",)

    novelty_failures = (
        (
            f"novelty {candidate.novelty:.6f} is below policy minimum "
            f"{policy.min_novelty:.6f}",
        )
        if candidate.novelty < policy.min_novelty
        else ()
    )

    missing_resources = _missing(
        candidate.resource_classes, policy.available_resource_classes
    )
    resource_failures = tuple(
        f"required resource class is unavailable: {item}"
        for item in missing_resources
    )
    if candidate.estimated_resource_cost > policy.max_estimated_resource_cost:
        resource_failures += (
            "estimated resource cost exceeds the frozen policy bound",
        )
    if candidate.estimated_tokens > policy.max_estimated_tokens:
        resource_failures += (
            "estimated token cost exceeds the frozen policy bound",
        )

    assumptions_total = len(candidate.assumptions) + len(
        candidate.semantic_requirements
    )
    assumption_score = (
        Decimal(1)
        if assumptions_total == 0
        else Decimal(
            assumptions_total
            - len(set(_casefold_set((*unvalidated, *untrusted))))
            - len(set(_casefold_set((*locally_unsupported, *policy_unsupported))))
        )
        / Decimal(assumptions_total)
    )
    dependency_score = (
        Decimal(1)
        if not candidate.dependencies
        else Decimal(len(candidate.dependencies) - len(unsatisfied))
        / Decimal(len(candidate.dependencies))
    )
    # Critical-path failures are retained explicitly even though they are also
    # a subset of dependency failures.  This makes scheduler diagnostics
    # actionable without double-counting the score.
    if critical_unsatisfied:
        dependency_failures += tuple(
            f"critical-path dependency is not ready: {item}"
            for item in critical_unsatisfied
        )
    cost_ratios = (
        Decimal(str(candidate.estimated_resource_cost))
        / Decimal(str(policy.max_estimated_resource_cost or 1)),
        Decimal(candidate.estimated_tokens)
        / Decimal(policy.max_estimated_tokens or 1),
    )
    cost_efficiency = Decimal(1) - min(Decimal(1), max(cost_ratios))

    dimensions = (
        _dimension(
            PlanEvaluationDimension.ACCEPTANCE_AND_EVIDENCE,
            failures=coverage_failures,
            successes=(
                "all frozen acceptance criteria and objective evidence terms are covered",
            ),
            hard_gate=True,
            score=coverage_score,
        ),
        _dimension(
            PlanEvaluationDimension.ASSUMPTIONS_AND_SEMANTICS,
            failures=assumption_failures,
            successes=(
                "all assumptions are validated and all required semantics are supported",
            ),
            hard_gate=True,
            score=max(Decimal(0), assumption_score),
        ),
        _dimension(
            PlanEvaluationDimension.DEPENDENCIES_AND_CRITICAL_PATH,
            failures=dependency_failures,
            successes=dependency_successes,
            hard_gate=True,
            score=dependency_score,
        ),
        _dimension(
            PlanEvaluationDimension.CONFLICT_SCOPE_AND_AUTHORITY,
            failures=conflict_failures,
            successes=(
                "changed scopes are authorized and have no unresolved conflicts",
            ),
            hard_gate=True,
            score=Decimal(0) if conflict_failures else Decimal(1),
        ),
        _dimension(
            PlanEvaluationDimension.VALIDATION_AND_PROOF,
            failures=validation_failures,
            successes=("validation and required proof are feasible",),
            hard_gate=True,
            score=Decimal(0) if validation_failures else Decimal(1),
        ),
        _dimension(
            PlanEvaluationDimension.NOVELTY,
            failures=novelty_failures,
            successes=(
                f"novelty is {candidate.novelty:.6f}",
            ),
            hard_gate=policy.min_novelty > 0,
            score=Decimal(str(candidate.novelty)),
        ),
        _dimension(
            PlanEvaluationDimension.RESOURCES_AND_TOKEN_COST,
            failures=resource_failures,
            successes=(
                "required resources are available",
                f"estimated token cost is {candidate.estimated_tokens}",
                "estimated resource cost is "
                f"{candidate.estimated_resource_cost:.6f}",
            ),
            hard_gate=True,
            score=cost_efficiency,
        ),
    )
    # Scores rank only already-admissible plans.  Hard-gate failures can never
    # be traded for cost, novelty, or any other high factor.
    score = sum(
        (Decimal(item.score_millionths) for item in dimensions),
        Decimal(0),
    ) / Decimal(len(dimensions))
    failures = tuple(
        item.dimension.value
        for item in dimensions
        if item.hard_gate and not item.passed
    )
    return EvaluatedEvidenceAwarePlan(
        candidate=candidate,
        score_millionths=int(score.quantize(Decimal("1"), rounding=ROUND_HALF_UP)),
        dimensions=dimensions,
        hard_gate_failures=failures,
    )


def evaluate_evidence_aware_plans(
    candidates: Iterable[EvidenceAwarePlanCandidate | Mapping[str, Any]],
    *,
    policy: EvidenceAwarePlanPolicy | Mapping[str, Any],
) -> EvidenceAwarePlanEvaluation:
    """Apply frozen-goal hard gates, then rank feasible plans deterministically.

    In particular, authority, scope, semantics, conflicts, proof feasibility,
    and finite resource bounds are non-compensable.  A cheaper unsafe plan is
    retained with diagnostics but can never outrank or replace a safe plan.
    """

    resolved_policy = (
        policy
        if isinstance(policy, EvidenceAwarePlanPolicy)
        else EvidenceAwarePlanPolicy.from_dict(policy)
    )
    normalized = tuple(
        item
        if isinstance(item, EvidenceAwarePlanCandidate)
        else EvidenceAwarePlanCandidate.from_dict(item)
        for item in candidates
    )
    if not normalized:
        raise ValueError("at least one evidence-aware plan candidate is required")
    candidate_ids = [item.candidate_id for item in normalized]
    duplicates = sorted(
        candidate_id
        for candidate_id in set(candidate_ids)
        if candidate_ids.count(candidate_id) > 1
    )
    if duplicates:
        raise ValueError(
            "evidence-aware candidate ids must be unique: "
            + ", ".join(duplicates)
        )
    evaluated = [_assess_evidence_aware_plan(item, resolved_policy) for item in normalized]
    admissible = sorted(
        (item for item in evaluated if item.admissible),
        key=lambda item: (
            -item.score_millionths,
            item.candidate_id,
            json.dumps(
                item.candidate.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
            ),
        ),
    )
    rejected = sorted(
        (item for item in evaluated if not item.admissible),
        key=lambda item: (
            len(item.hard_gate_failures),
            -item.score_millionths,
            item.candidate_id,
        ),
    )
    return EvidenceAwarePlanEvaluation(
        selected=admissible[0] if admissible else None,
        admissible=tuple(admissible),
        rejected=tuple(rejected),
        policy=resolved_policy,
    )


def validate_evidence_aware_plan_evaluation(
    evaluation: EvidenceAwarePlanEvaluation,
) -> EvidenceAwarePlanEvaluation:
    """Recompute and validate a persisted deterministic plan evaluation.

    Dataclass invariants prove that a serialized evaluation is structurally
    coherent, but structure alone cannot prove its scores, diagnostics, or
    winner were produced by the current evaluator.  Receipt consumers call
    this boundary before trusting a restored selection.  The candidate order
    is intentionally irrelevant because :func:`evaluate_evidence_aware_plans`
    applies the canonical ranking and tie-break rules.
    """

    if not isinstance(evaluation, EvidenceAwarePlanEvaluation):
        raise PlanBranchValidationError(
            "evaluation must be EvidenceAwarePlanEvaluation"
        )
    recomputed = evaluate_evidence_aware_plans(
        (item.candidate for item in evaluation.ranked),
        policy=evaluation.policy,
    )
    if recomputed != evaluation:
        raise PlanBranchValidationError(
            "evidence-aware plan evaluation does not match deterministic recomputation"
        )
    return evaluation


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


# Bounded objective-work evaluation ------------------------------------------------


def _proposal_value(proposal: object, *names: str, default: Any = None) -> Any:
    """Read one canonical work-proposal field without importing objective_graph.

    ``objective_graph`` owns the proposal model and uses this evaluator.  Keeping
    the boundary structural avoids an import cycle and also lets persisted JSON
    proposals be re-evaluated before they are materialized.
    """

    for name in names:
        if isinstance(proposal, Mapping) and name in proposal:
            value = proposal[name]
            if value not in (None, ""):
                return value
        if hasattr(proposal, name):
            value = getattr(proposal, name)
            if value not in (None, ""):
                return value
    return default


def _proposal_strings(
    proposal: object,
    *names: str,
    required: bool = False,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    present = any(
        (isinstance(proposal, Mapping) and name in proposal)
        or (not isinstance(proposal, Mapping) and hasattr(proposal, name))
        for name in names
    )
    value = _proposal_value(proposal, *names)
    field_name = names[0]
    if required and not present:
        raise PlanBranchValidationError(
            f"objective work proposal {field_name} must be recorded"
        )
    if value in (None, ""):
        if required and not allow_empty:
            raise PlanBranchValidationError(
                f"objective work proposal {field_name} must contain at least one value"
            )
        return ()
    if isinstance(value, str):
        values: Sequence[Any] = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = value
    else:
        raise PlanBranchValidationError(
            f"objective work proposal {field_name} must be a string or array of strings"
        )
    return _string_tuple(
        values,
        f"objective work proposal {field_name}",
        allow_empty=allow_empty or not required,
    )


def _normalized_semantic_value(value: str) -> str:
    return " ".join(value.casefold().split())


def _objective_work_payload(proposal: object) -> dict[str, Any]:
    if isinstance(proposal, Mapping):
        return dict(proposal)
    to_dict = getattr(proposal, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return dict(payload)
    if hasattr(proposal, "__dataclass_fields__"):
        return asdict(proposal)
    raise PlanBranchValidationError(
        "objective work proposal must be a mapping, dataclass, or expose to_dict()"
    )


def _profile_g_objective_work_payload(proposal: object) -> dict[str, Any]:
    """Encode proposal metrics without emitting Profile G-incompatible floats."""

    payload = _objective_work_payload(proposal)
    for name in ("confidence", "novelty", "estimated_cost", "cost"):
        value = payload.pop(name, None)
        if value is not None:
            payload[f"{name}_millionths"] = _to_millionths(value)
    return payload


@dataclass(frozen=True)
class ObjectiveWorkEvaluationPolicy:
    """Finite, scheduler-aware admission policy for generated objective work."""

    min_confidence: float = 0.65
    min_novelty: float = 0.35
    max_proposals: int = 5
    max_total_cost: float = 20.0
    max_open_work: int = 20
    current_open_work: int = 0
    remaining_token_budget: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "min_confidence",
            _number(self.min_confidence, "min_confidence", minimum=0.0, maximum=1.0),
        )
        object.__setattr__(
            self,
            "min_novelty",
            _number(self.min_novelty, "min_novelty", minimum=0.0, maximum=1.0),
        )
        object.__setattr__(
            self,
            "max_total_cost",
            _number(self.max_total_cost, "max_total_cost", minimum=0.0),
        )
        for name in ("max_proposals", "max_open_work", "current_open_work"):
            raw = getattr(self, name)
            if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.remaining_token_budget is not None:
            raw_tokens = self.remaining_token_budget
            if (
                isinstance(raw_tokens, bool)
                or not isinstance(raw_tokens, int)
                or raw_tokens < 0
            ):
                raise ValueError("remaining_token_budget must be a non-negative integer or None")

    @property
    def available_open_slots(self) -> int:
        return max(0, self.max_open_work - self.current_open_work)

    @property
    def admission_limit(self) -> int:
        return min(self.max_proposals, self.available_open_slots)

    def to_dict(self, *, profile_g: bool = False) -> dict[str, Any]:
        result = asdict(self)
        if profile_g:
            result["min_confidence_millionths"] = _to_millionths(
                result.pop("min_confidence")
            )
            result["min_novelty_millionths"] = _to_millionths(
                result.pop("min_novelty")
            )
            result["max_total_cost_millionths"] = _to_millionths(
                result.pop("max_total_cost")
            )
        return result


@dataclass(frozen=True)
class EvaluatedObjectiveWorkProposal:
    """One generated work proposal with its reproducible admission score."""

    proposal: object
    canonical_id: str
    semantic_key: str
    score_millionths: int
    rationale: tuple[str, ...]

    def to_dict(self, *, profile_g: bool = False) -> dict[str, Any]:
        return {
            "proposal": (
                _profile_g_objective_work_payload(self.proposal)
                if profile_g
                else _objective_work_payload(self.proposal)
            ),
            "canonical_id": self.canonical_id,
            "semantic_key": self.semantic_key,
            "score_millionths": int(self.score_millionths),
            "rationale": list(self.rationale),
        }


@dataclass(frozen=True)
class RejectedObjectiveWorkProposal:
    """One valid generated proposal excluded by a documented policy rule."""

    evaluated: EvaluatedObjectiveWorkProposal
    reason: str

    @property
    def proposal(self) -> object:
        return self.evaluated.proposal

    @property
    def canonical_id(self) -> str:
        return self.evaluated.canonical_id

    @property
    def semantic_key(self) -> str:
        return self.evaluated.semantic_key

    def to_dict(self, *, profile_g: bool = False) -> dict[str, Any]:
        return {
            **self.evaluated.to_dict(profile_g=profile_g),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ObjectiveWorkProposalEvaluation:
    """Auditable admission result for one bounded autonomous-refinement cycle."""

    accepted: tuple[EvaluatedObjectiveWorkProposal, ...]
    rejected: tuple[RejectedObjectiveWorkProposal, ...]
    policy: ObjectiveWorkEvaluationPolicy
    admitted_cost: float
    admitted_tokens: int
    evaluator_version: str = OBJECTIVE_WORK_EVALUATOR_VERSION

    @property
    def selected(self) -> object | None:
        return self.accepted[0].proposal if self.accepted else None

    @property
    def accepted_proposals(self) -> tuple[object, ...]:
        return tuple(item.proposal for item in self.accepted)

    @property
    def remaining_open_slots(self) -> int:
        return max(0, self.policy.available_open_slots - len(self.accepted))

    @property
    def remaining_cost(self) -> float:
        return max(0.0, self.policy.max_total_cost - self.admitted_cost)

    @property
    def remaining_tokens(self) -> int | None:
        if self.policy.remaining_token_budget is None:
            return None
        return max(0, self.policy.remaining_token_budget - self.admitted_tokens)

    def to_dict(self, *, profile_g: bool = False) -> dict[str, Any]:
        admitted_cost_key = "admitted_cost_millionths" if profile_g else "admitted_cost"
        remaining_cost_key = "remaining_cost_millionths" if profile_g else "remaining_cost"
        return {
            "evaluator_version": self.evaluator_version,
            "accepted": [item.to_dict(profile_g=profile_g) for item in self.accepted],
            "rejected": [item.to_dict(profile_g=profile_g) for item in self.rejected],
            "policy": self.policy.to_dict(profile_g=profile_g),
            admitted_cost_key: (
                _to_millionths(self.admitted_cost) if profile_g else self.admitted_cost
            ),
            remaining_cost_key: (
                _to_millionths(self.remaining_cost) if profile_g else self.remaining_cost
            ),
            "admitted_tokens": self.admitted_tokens,
            "remaining_tokens": self.remaining_tokens,
            "remaining_open_slots": self.remaining_open_slots,
        }

    def to_profile_g_dict(self) -> dict[str, Any]:
        return self.to_dict(profile_g=True)


@dataclass(frozen=True)
class _ObjectiveWorkCandidate:
    proposal: object
    canonical_id: str
    semantic_key: str
    parent_objective_terms: tuple[str, ...]
    expected_evidence_delta: tuple[str, ...]
    confidence: float
    novelty: float
    cost: float
    estimated_tokens: int
    evaluated: EvaluatedObjectiveWorkProposal


def _objective_work_candidate(proposal: object) -> _ObjectiveWorkCandidate:
    # Force serialization here so unsupported mutable/runtime-only values fail
    # before any proposal can consume scheduler capacity.
    _objective_work_payload(proposal)
    canonical_id = _required_string(
        _proposal_value(
            proposal,
            "canonical_id",
            "canonical_identity",
            "canonical_task_cid",
            "proposal_id",
        ),
        "objective work proposal canonical_id",
    )
    semantic_key = _required_string(
        _proposal_value(proposal, "semantic_key", "semantic_identity"),
        "objective work proposal semantic_key",
    )
    terms = _proposal_strings(
        proposal,
        "parent_objective_terms",
        "objective_terms",
        required=True,
    )
    evidence_delta = _proposal_strings(
        proposal,
        "expected_evidence_delta",
        "evidence_delta",
        required=True,
    )
    # These fields are deliberately read even though only their presence affects
    # admission.  Scheduler work that omits execution/proof surfaces is not a
    # complete objective-work proposal.
    _proposal_strings(proposal, "dependencies", "depends_on")
    _proposal_strings(
        proposal, "predicted_files", "files", required=True, allow_empty=True
    )
    _proposal_strings(
        proposal,
        "predicted_symbols",
        "symbols",
        "ast_symbols",
        required=True,
        allow_empty=True,
    )
    _proposal_strings(
        proposal,
        "validation_commands",
        "validation",
        required=True,
        allow_empty=True,
    )
    confidence = _number(
        _proposal_value(proposal, "confidence"),
        "objective work proposal confidence",
        minimum=0.0,
        maximum=1.0,
    )
    novelty = _number(
        _proposal_value(proposal, "novelty"),
        "objective work proposal novelty",
        minimum=0.0,
        maximum=1.0,
    )
    cost = _number(
        _proposal_value(proposal, "cost", "estimated_cost"),
        "objective work proposal cost",
        minimum=0.0,
    )
    raw_tokens = _proposal_value(
        proposal,
        "estimated_tokens",
        "token_cost",
        default=0,
    )
    if isinstance(raw_tokens, bool) or not isinstance(raw_tokens, int) or raw_tokens < 0:
        raise PlanBranchValidationError(
            "objective work proposal estimated_tokens must be a non-negative integer"
        )
    confidence_decimal = Decimal(str(confidence))
    novelty_decimal = Decimal(str(novelty))
    cost_efficiency = Decimal(1) / (Decimal(1) + Decimal(str(cost)))
    evidence_specificity = min(Decimal(1), Decimal(len(evidence_delta)) / Decimal(2))
    score = (
        Decimal("0.45") * confidence_decimal
        + Decimal("0.25") * novelty_decimal
        + Decimal("0.20") * cost_efficiency
        + Decimal("0.10") * evidence_specificity
    )
    score_millionths = _to_millionths(score)
    rationale = (
        f"confidence contributes {_to_millionths(Decimal('0.45') * confidence_decimal)} millionths",
        f"novelty contributes {_to_millionths(Decimal('0.25') * novelty_decimal)} millionths",
        f"cost efficiency contributes {_to_millionths(Decimal('0.20') * cost_efficiency)} millionths",
        f"expected evidence specificity contributes {_to_millionths(Decimal('0.10') * evidence_specificity)} millionths",
        f"total deterministic admission score is {score_millionths} millionths",
    )
    evaluated = EvaluatedObjectiveWorkProposal(
        proposal=proposal,
        canonical_id=canonical_id,
        semantic_key=semantic_key,
        score_millionths=score_millionths,
        rationale=rationale,
    )
    return _ObjectiveWorkCandidate(
        proposal=proposal,
        canonical_id=canonical_id,
        semantic_key=semantic_key,
        parent_objective_terms=terms,
        expected_evidence_delta=evidence_delta,
        confidence=confidence,
        novelty=novelty,
        cost=cost,
        estimated_tokens=raw_tokens,
        evaluated=evaluated,
    )


def evaluate_objective_work_proposals(
    proposals: Iterable[object],
    *,
    policy: ObjectiveWorkEvaluationPolicy | None = None,
    objective_terms: Sequence[str] = (),
    known_canonical_ids: Iterable[str] = (),
    known_semantic_keys: Iterable[str] = (),
) -> ObjectiveWorkProposalEvaluation:
    """Validate, deduplicate, rank, and bound generated goals/subgoals/tasks.

    Candidate order never affects the result.  Higher-scoring equivalents win
    semantic deduplication; capacity is then reserved in score order so a router
    cannot crowd out stronger work by placing a weak proposal first.
    """

    resolved_policy = policy or ObjectiveWorkEvaluationPolicy()
    required_terms = {
        _normalized_semantic_value(str(term))
        for term in objective_terms
        if str(term).strip()
    }
    known_ids = {str(item).strip() for item in known_canonical_ids if str(item).strip()}
    known_semantics = {
        _normalized_semantic_value(str(item))
        for item in known_semantic_keys
        if str(item).strip()
    }
    candidates = [_objective_work_candidate(proposal) for proposal in proposals]
    candidates.sort(
        key=lambda item: (
            -item.evaluated.score_millionths,
            item.canonical_id,
            _normalized_semantic_value(item.semantic_key),
            json.dumps(
                _objective_work_payload(item.proposal),
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ),
        )
    )
    accepted: list[EvaluatedObjectiveWorkProposal] = []
    rejected: list[RejectedObjectiveWorkProposal] = []
    observed_ids = set(known_ids)
    observed_semantics = set(known_semantics)
    admitted_cost = Decimal(0)
    admitted_tokens = 0
    cost_limit = Decimal(str(resolved_policy.max_total_cost))

    for candidate in candidates:
        normalized_semantic = _normalized_semantic_value(candidate.semantic_key)
        candidate_terms = {
            _normalized_semantic_value(term)
            for term in candidate.parent_objective_terms
        }
        work_kind = _normalized_semantic_value(
            str(_proposal_value(candidate.proposal, "kind", "work_kind", default="task"))
        )
        reason = ""
        if candidate.canonical_id in observed_ids:
            reason = "duplicate_canonical_identity"
        elif normalized_semantic in observed_semantics:
            reason = "duplicate_semantic_work"
        else:
            # Reserve identities before policy filters. Rephrasing a rejected
            # equivalent within the same cycle must not receive another chance.
            observed_ids.add(candidate.canonical_id)
            observed_semantics.add(normalized_semantic)
            if (
                work_kind != "goal"
                and required_terms
                and not (required_terms & candidate_terms)
            ):
                reason = "no_parent_objective_term_coverage"
            elif candidate.confidence < resolved_policy.min_confidence:
                reason = "confidence_below_threshold"
            elif candidate.novelty < resolved_policy.min_novelty:
                reason = "novelty_below_threshold"
            elif len(accepted) >= resolved_policy.available_open_slots:
                reason = "open_work_limit_reached"
            elif len(accepted) >= resolved_policy.max_proposals:
                reason = "proposal_limit_reached"
            elif admitted_cost + Decimal(str(candidate.cost)) > cost_limit:
                reason = "cost_limit_reached"
            elif (
                resolved_policy.remaining_token_budget is not None
                and admitted_tokens + candidate.estimated_tokens
                > resolved_policy.remaining_token_budget
            ):
                reason = "token_limit_reached"

        if reason:
            rejected.append(RejectedObjectiveWorkProposal(candidate.evaluated, reason))
            continue
        accepted.append(candidate.evaluated)
        admitted_cost += Decimal(str(candidate.cost))
        admitted_tokens += candidate.estimated_tokens

    rejected.sort(
        key=lambda item: (
            item.reason,
            -item.evaluated.score_millionths,
            item.canonical_id,
        )
    )
    return ObjectiveWorkProposalEvaluation(
        accepted=tuple(accepted),
        rejected=tuple(rejected),
        policy=resolved_policy,
        admitted_cost=float(admitted_cost),
        admitted_tokens=admitted_tokens,
    )
