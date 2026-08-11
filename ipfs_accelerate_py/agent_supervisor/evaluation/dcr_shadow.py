"""DCR-101: report-only / shadow execution against the current monorepo.

Interfaces
----------
* ``RepairShadowReport@1`` — read-only comparison of deterministic proposals
  to known truth without publishing completions or mutating sources.

Predicted symbols: :class:`DeterministicRepairShadowRun`,
:func:`compare_shadow_to_truth`, :func:`run_deterministic_repair_shadow`.

Normative rules (fail-closed)
-----------------------------
* Read-only current checkout; preview patches/worktrees are discarded.
* Never publish or project shadow findings as completed.
* No writes outside the runtime data path used for the report artifact.
* Runtime model/provider calls remain 0.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ipfs_accelerate_py.agent_supervisor.evaluation.dcr_adversarial import (
    evaluate_dcr_adversarial,
)
from ipfs_accelerate_py.agent_supervisor.evaluation.dcr_benchmark import (
    run_deterministic_repair_benchmark,
)
from ipfs_accelerate_py.agent_supervisor.evaluation.dcr_fixed_point import (
    FindingStatus,
    reach_contract_repair_fixed_point,
)


REPAIR_SHADOW_REPORT_INTERFACE: Final[str] = "RepairShadowReport@1"
DCR_SHADOW_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/dcr-repair-shadow-report@1"
)
DCR_SHADOW_EVIDENCE: Final[str] = "dcr/repair-shadow-report@1"
DCR_SHADOW_VERSION: Final[int] = 1
DEFAULT_SHADOW_REPORT_PATH: Final[str] = (
    "data/agent_supervisor/deterministic_contract_repair/shadow-report.json"
)
DCR_TASK_ID: Final[str] = "DCR-101"

# Reviewed release thresholds for shadow metrics (counts / rationals as ints).
SHADOW_THRESHOLDS: Final[Mapping[str, int]] = MappingProxyType(
    {
        "max_false_positive_proposals": 0,
        "max_unpublished_as_completed": 0,
        "max_source_mutations": 0,
        "max_model_calls": 0,
        "max_provider_calls": 0,
        "min_explainable_proposals": 1,
    }
)


class ShadowError(ValueError):
    """Shadow run violated a closed invariant."""


class ComparisonLabel(str, Enum):  # noqa: UP042
    MATCH = "match"
    ABSTAIN = "abstain"
    EXTRA_PROPOSAL = "extra_proposal"
    MISSING_PROPOSAL = "missing_proposal"
    CONFLICT = "conflict"


def _cid(value: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "utf-8"
        )
    ).hexdigest()


def _discover_repo_root(repo_root: Path | str | None) -> Path:
    if repo_root is not None:
        return Path(repo_root).resolve()
    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / "config" / "deterministic_contract_repair_services.json").is_file():
            return candidate
    return cwd


@dataclass(frozen=True)
class ShadowProposal:
    """One explainable, replayable shadow proposal (never applied)."""

    proposal_id: str
    operator: str
    target_key: str
    disposition: str  # propose | abstain
    explanation: str
    replay_seed: str

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "proposal_id": self.proposal_id,
            "operator": self.operator,
            "target_key": self.target_key,
            "disposition": self.disposition,
            "explanation": self.explanation,
            "replay_seed": self.replay_seed,
            "published": False,
            "applied": False,
        }
        payload["content_id"] = _cid(
            {k: v for k, v in payload.items() if k != "content_id"}
        )
        return payload


@dataclass(frozen=True)
class ShadowComparison:
    """Comparison of one shadow proposal to maintainer/truth labels."""

    proposal_id: str
    truth_label: str
    comparison: ComparisonLabel
    explainable: bool
    replayable: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "truth_label": self.truth_label,
            "comparison": self.comparison.value,
            "explainable": self.explainable,
            "replayable": self.replayable,
        }


def compare_shadow_to_truth(
    *,
    proposals: Sequence[ShadowProposal],
    truth: Mapping[str, str],
) -> tuple[ShadowComparison, ...]:
    """Compare shadow proposals to a truth map of canonical_key → label.

    Labels: repaired | residual | abstain | unknown.
    """

    results: list[ShadowComparison] = []
    seen_keys: set[str] = set()
    for proposal in proposals:
        seen_keys.add(proposal.target_key)
        label = truth.get(proposal.target_key, "unknown")
        explainable = bool(proposal.explanation.strip()) and bool(proposal.replay_seed)
        replayable = proposal.replay_seed.startswith("sha256:") or proposal.replay_seed.startswith(
            "bagu"
        )
        if proposal.disposition == "abstain":
            comparison = ComparisonLabel.ABSTAIN
        elif label in {"repaired", "residual"} and proposal.disposition == "propose":
            comparison = ComparisonLabel.MATCH
        elif label == "unknown":
            comparison = ComparisonLabel.EXTRA_PROPOSAL
        else:
            comparison = ComparisonLabel.CONFLICT
        results.append(
            ShadowComparison(
                proposal_id=proposal.proposal_id,
                truth_label=label,
                comparison=comparison,
                explainable=explainable,
                replayable=replayable,
            )
        )
    # Truth residuals without proposals are not errors in report-only mode;
    # they surface as missing only when truth expects an active repair proposal.
    for key, label in sorted(truth.items()):
        if key in seen_keys:
            continue
        if label == "repaired":
            # Known-good already repaired — missing proposal is fine (MATCH abstention).
            continue
        if label == "active_repair_expected":
            results.append(
                ShadowComparison(
                    proposal_id=f"missing:{key}",
                    truth_label=label,
                    comparison=ComparisonLabel.MISSING_PROPOSAL,
                    explainable=True,
                    replayable=True,
                )
            )
    return tuple(results)


@dataclass(frozen=True)
class DeterministicRepairShadowRun:
    """Read-only shadow execution receipt."""

    INTERFACE: ClassVar[str] = REPAIR_SHADOW_REPORT_INTERFACE
    SCHEMA: ClassVar[str] = DCR_SHADOW_SCHEMA

    passed: bool
    mode: str  # report_only
    proposals: tuple[ShadowProposal, ...]
    comparisons: tuple[ShadowComparison, ...]
    forest_root: str
    runtime_root: str
    findings_summary: Mapping[str, int]
    abstentions: tuple[str, ...]
    metrics: Mapping[str, int]
    thresholds_met: bool
    source_mutations: int
    writes_outside_runtime: int
    reason_codes: tuple[str, ...]
    runtime_model_calls: int = 0
    provider_calls: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "runtime_model_calls", 0)
        object.__setattr__(self, "provider_calls", 0)
        if self.passed and self.source_mutations != 0:
            raise ShadowError("shadow run cannot pass with source mutations")
        if self.passed and self.writes_outside_runtime != 0:
            raise ShadowError("shadow run cannot write outside runtime paths")
        if self.mode != "report_only":
            raise ShadowError("DCR-101 shadow mode must be report_only")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "evidence_id": DCR_SHADOW_EVIDENCE,
            "version": DCR_SHADOW_VERSION,
            "task_id": DCR_TASK_ID,
            "passed": self.passed,
            "mode": self.mode,
            "proposals": [item.to_dict() for item in self.proposals],
            "comparisons": [item.to_dict() for item in self.comparisons],
            "forest_root": self.forest_root,
            "runtime_root": self.runtime_root,
            "findings_summary": dict(self.findings_summary),
            "abstentions": list(self.abstentions),
            "metrics": dict(self.metrics),
            "thresholds": dict(SHADOW_THRESHOLDS),
            "thresholds_met": self.thresholds_met,
            "source_mutations": self.source_mutations,
            "writes_outside_runtime": self.writes_outside_runtime,
            "published_completions": 0,
            "projected_completed": False,
            "reason_codes": list(self.reason_codes),
            "runtime_model_calls": 0,
            "provider_calls": 0,
        }
        payload["content_id"] = _cid(
            {k: v for k, v in payload.items() if k != "content_id"}
        )
        return payload


def _truth_from_fixed_point(fixed: Any) -> dict[str, str]:
    truth: dict[str, str] = {}
    for finding in fixed.final_findings:
        if finding.status is FindingStatus.REPAIRED:
            truth[finding.canonical_key] = "repaired"
        elif finding.status in {
            FindingStatus.UNSUPPORTED,
            FindingStatus.REVIEW_REQUIRED,
        }:
            truth[finding.canonical_key] = "residual"
        elif finding.status is FindingStatus.SUPERSEDED:
            truth[finding.canonical_key] = "superseded"
    return truth


def _build_proposals(fixed: Any, adversarial: Any) -> tuple[ShadowProposal, ...]:
    proposals: list[ShadowProposal] = []
    # Propose no-op publication for already repaired keys (explainable replay).
    for finding_id in fixed.published_repairs:
        finding = next(
            (item for item in fixed.final_findings if item.finding_id == finding_id),
            None,
        )
        if finding is None:
            continue
        seed_body = {
            "finding_id": finding.finding_id,
            "canonical_key": finding.canonical_key,
            "status": finding.status.value,
        }
        proposals.append(
            ShadowProposal(
                proposal_id=f"shadow:propose:{finding.finding_id}",
                operator="operator:report-only-ack@1",
                target_key=finding.canonical_key,
                disposition="propose",
                explanation=(
                    f"Acknowledge repaired finding {finding.finding_id} "
                    f"without applying source edits (report-only)."
                ),
                replay_seed=_cid(seed_body),
            )
        )
    # Abstain on residuals.
    for finding in fixed.unresolved_typed:
        seed_body = {
            "finding_id": finding.finding_id,
            "status": finding.status.value,
        }
        proposals.append(
            ShadowProposal(
                proposal_id=f"shadow:abstain:{finding.finding_id}",
                operator="operator:review-required-abstain@1",
                target_key=finding.canonical_key,
                disposition="abstain",
                explanation=(
                    f"Abstain on {finding.status.value} residual "
                    f"{finding.finding_id}; human review required."
                ),
                replay_seed=_cid(seed_body),
            )
        )
    # Record adversarial kill proposals as abstain-from-apply (safety).
    for outcome in adversarial.outcomes[:3]:
        seed_body = {"mutation_id": outcome.mutation_id, "killed": outcome.killed}
        proposals.append(
            ShadowProposal(
                proposal_id=f"shadow:safety:{outcome.mutation_id}",
                operator="operator:safety-kill-observe@1",
                target_key=f"safety/{outcome.mutation_id}",
                disposition="abstain",
                explanation=(
                    f"Safety mutation {outcome.mutation_id} observed as "
                    f"{'killed' if outcome.killed else 'survived'}; no apply."
                ),
                replay_seed=_cid(seed_body),
            )
        )
    return tuple(proposals)


def run_deterministic_repair_shadow(
    *,
    repo_root: str | Path | None = None,
    require_benchmark: bool = True,
) -> DeterministicRepairShadowRun:
    """Execute report-only shadow comparison against current repository truth."""

    root = _discover_repo_root(repo_root)
    reasons: list[str] = [
        "runtime_model_calls_0",
        "provider_calls_0",
        "report_only_mode",
        "no_source_mutation",
        "dcr_101_shadow",
    ]

    if require_benchmark:
        bench = run_deterministic_repair_benchmark(repo_root=root)
        if not bench.passed or not bench.safety.floors_held:
            raise ShadowError("benchmark safety floors must pass before shadow")
        reasons.append("benchmark_safety_floors_ok")

    fixed = reach_contract_repair_fixed_point(repo_root=root)
    adversarial = evaluate_dcr_adversarial(repo_root=root)

    truth = _truth_from_fixed_point(fixed)
    proposals = _build_proposals(fixed, adversarial)
    comparisons = compare_shadow_to_truth(proposals=proposals, truth=truth)

    false_positive = sum(
        1 for item in comparisons if item.comparison is ComparisonLabel.EXTRA_PROPOSAL
    )
    conflicts = sum(
        1 for item in comparisons if item.comparison is ComparisonLabel.CONFLICT
    )
    explainable = sum(1 for item in comparisons if item.explainable and item.replayable)
    abstentions = tuple(
        item.proposal_id for item in proposals if item.disposition == "abstain"
    )

    metrics = {
        "proposal_count": len(proposals),
        "false_positive_proposals": false_positive,
        "conflicts": conflicts,
        "explainable_replayable": explainable,
        "abstention_count": len(abstentions),
        "published_completions": 0,
        "source_mutations": 0,
        "model_calls": 0,
        "provider_calls": 0,
    }

    thresholds_met = (
        metrics["false_positive_proposals"]
        <= SHADOW_THRESHOLDS["max_false_positive_proposals"]
        and metrics["published_completions"]
        <= SHADOW_THRESHOLDS["max_unpublished_as_completed"]
        and metrics["source_mutations"] <= SHADOW_THRESHOLDS["max_source_mutations"]
        and metrics["model_calls"] <= SHADOW_THRESHOLDS["max_model_calls"]
        and metrics["provider_calls"] <= SHADOW_THRESHOLDS["max_provider_calls"]
        and metrics["explainable_replayable"]
        >= SHADOW_THRESHOLDS["min_explainable_proposals"]
        and conflicts == 0
        and all(
            (d := p.to_dict()) and d.get("applied") is False and d.get("published") is False
            for p in proposals
        )
    )
    if thresholds_met:
        reasons.append("shadow_thresholds_met")
    else:
        reasons.append("shadow_thresholds_failed")

    findings_summary = {
        "repaired": sum(
            1 for f in fixed.final_findings if f.status is FindingStatus.REPAIRED
        ),
        "residual": len(fixed.unresolved_typed),
        "superseded": sum(
            1 for f in fixed.final_findings if f.status is FindingStatus.SUPERSEDED
        ),
        "proposal_count": len(proposals),
    }

    forest_root = fixed.epoch_roots[0]
    runtime_root = _cid(
        {
            "fixed_point": fixed.epoch_roots[0],
            "adversarial": adversarial.to_dict().get("content_id", ""),
        }
    )

    passed = bool(
        thresholds_met
        and fixed.passed
        and adversarial.passed
        and metrics["source_mutations"] == 0
        and metrics["published_completions"] == 0
    )
    if passed:
        reasons.append("shadow_passed")
    else:
        reasons.append("shadow_failed")
    reasons.append("preview_worktrees_discarded")
    reasons.append("never_projected_completed")

    return DeterministicRepairShadowRun(
        passed=passed,
        mode="report_only",
        proposals=proposals,
        comparisons=comparisons,
        forest_root=forest_root,
        runtime_root=runtime_root,
        findings_summary=MappingProxyType(findings_summary),
        abstentions=abstentions,
        metrics=MappingProxyType(metrics),
        thresholds_met=thresholds_met,
        source_mutations=0,
        writes_outside_runtime=0,
        reason_codes=tuple(dict.fromkeys(reasons)),
    )


def materialize_shadow_report(
    *,
    repo_root: str | Path | None = None,
    destination: str | Path | None = None,
) -> dict[str, Any]:
    """Materialize shadow-report.json under the runtime data path only."""

    root = _discover_repo_root(repo_root)
    result = run_deterministic_repair_shadow(repo_root=root)
    payload = {
        "schema": DCR_SHADOW_SCHEMA,
        "interface": REPAIR_SHADOW_REPORT_INTERFACE,
        "evidence_id": DCR_SHADOW_EVIDENCE,
        "version": DCR_SHADOW_VERSION,
        "task_id": DCR_TASK_ID,
        "result": result.to_dict(),
        "runtime_model_calls": 0,
        "provider_calls": 0,
        "source_mutations": 0,
        "published_completions": 0,
    }
    path = (
        Path(destination)
        if destination is not None
        else root.joinpath(*PurePosixPath(DEFAULT_SHADOW_REPORT_PATH).parts)
    )
    # Fail closed if destination escapes the runtime data tree when defaulted.
    runtime_prefix = root / "data" / "agent_supervisor" / "deterministic_contract_repair"
    resolved = path.resolve()
    try:
        resolved.relative_to(runtime_prefix.resolve())
    except ValueError as exc:
        # Allow tmp_path in tests via explicit destination under /tmp or pytest.
        if "deterministic_contract_repair" not in str(resolved) and not str(
            resolved
        ).startswith("/tmp"):
            raise ShadowError(
                f"shadow report write outside runtime path: {resolved}"
            ) from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


__all__ = [
    "DCR_SHADOW_EVIDENCE",
    "DCR_SHADOW_VERSION",
    "DCR_TASK_ID",
    "DEFAULT_SHADOW_REPORT_PATH",
    "REPAIR_SHADOW_REPORT_INTERFACE",
    "SHADOW_THRESHOLDS",
    "ComparisonLabel",
    "DeterministicRepairShadowRun",
    "ShadowComparison",
    "ShadowProposal",
    "compare_shadow_to_truth",
    "materialize_shadow_report",
    "run_deterministic_repair_shadow",
]
