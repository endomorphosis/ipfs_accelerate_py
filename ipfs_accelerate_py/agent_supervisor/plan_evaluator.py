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
PROOF_AWARE_PLAN_EVALUATOR_VERSION = "proof-aware-plan-evaluator-v1"
OBJECTIVE_WORK_EVALUATOR_VERSION = "objective-work-evaluator-v1"
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
