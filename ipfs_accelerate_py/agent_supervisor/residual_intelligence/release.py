"""Current-tree residual gap/release reporting. Reports cannot promote."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from .contracts import ResidualIntelligenceError, required_text

REPORT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-intelligence-release-report@1"
)
FORBIDDEN_CLAIMS: Final[frozenset[str]] = frozenset(
    {
        "learned",
        "verified",
        "safe",
        "autonomous",
        "token-efficient",
        "production-ready",
    }
)


@dataclass(frozen=True)
class ResidualGapReport:
    blockers: tuple[str, ...]
    unsupported_claims: tuple[str, ...]
    not_run: tuple[str, ...]


@dataclass(frozen=True)
class ResidualIntelligenceReleaseReport:
    start_tree: str
    end_tree: str
    corpus_admission_id: str
    expert_dispositions: Mapping[str, str]
    before: Mapping[str, int]
    after: Mapping[str, int]
    costs: Mapping[str, int]
    promotion_eligible: bool
    rollback_target: str
    gaps: ResidualGapReport
    schema: str = REPORT_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "start_tree", required_text(self.start_tree, "start_tree"))
        object.__setattr__(self, "end_tree", required_text(self.end_tree, "end_tree"))
        object.__setattr__(
            self,
            "corpus_admission_id",
            required_text(self.corpus_admission_id, "corpus_admission_id"),
        )
        object.__setattr__(
            self, "rollback_target", required_text(self.rollback_target, "rollback_target")
        )
        if type(self.promotion_eligible) is not bool:
            raise ResidualIntelligenceError("promotion_eligible must be boolean")
        if self.promotion_eligible:
            raise ResidualIntelligenceError("release reports cannot promote")
        unsupported = tuple(
            claim
            for claim in self.gaps.unsupported_claims
            if claim in FORBIDDEN_CLAIMS or True
        )
        object.__setattr__(
            self,
            "gaps",
            ResidualGapReport(
                blockers=tuple(self.gaps.blockers),
                unsupported_claims=tuple(self.gaps.unsupported_claims),
                not_run=tuple(self.gaps.not_run),
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "start_tree": self.start_tree,
            "end_tree": self.end_tree,
            "corpus_admission_id": self.corpus_admission_id,
            "expert_dispositions": dict(self.expert_dispositions),
            "before": dict(self.before),
            "after": dict(self.after),
            "costs": dict(self.costs),
            "promotion_eligible": False,
            "rollback_target": self.rollback_target,
            "gaps": {
                "blockers": self.gaps.blockers,
                "unsupported_claims": self.gaps.unsupported_claims,
                "not_run": self.gaps.not_run,
            },
        }


def validate_release_claims(report: ResidualIntelligenceReleaseReport) -> ResidualIntelligenceReleaseReport:
    for claim in report.gaps.unsupported_claims:
        if claim not in FORBIDDEN_CLAIMS:
            raise ResidualIntelligenceError(f"unknown unsupported claim token: {claim}")
    if report.promotion_eligible:
        raise ResidualIntelligenceError("release reports cannot promote")
    return report
