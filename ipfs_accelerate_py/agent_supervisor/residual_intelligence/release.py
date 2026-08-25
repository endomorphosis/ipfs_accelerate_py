"""Strict, non-authoritative final-tree release reporting.

This module validates a report *about* an inspected tree.  It does not inspect
Git, execute a benchmark, create a checkpoint, promote an expert, accept a
proof, or complete a goal.  Those effects remain with their existing owners.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final

from .contracts import ResidualIntelligenceError, bounded_json_mapping, required_text


REPORT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-intelligence-release-report@2"
)
FORBIDDEN_CLAIMS: Final[frozenset[str]] = frozenset(
    {
        "learned", "verified", "safe", "autonomous", "token-efficient", "production-ready",
    }
)
_TREE_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{40}$")
_REQUIRED_SECTION_NAMES: Final[frozenset[str]] = frozenset(
    {
        "lineage", "files_and_symbols", "corpus_rights_splits",
        "architecture_tokenizer_checkpoint", "metrics", "proof_validation", "drift",
        "promotion", "rollback", "report_authority",
    }
)
_DISPOSITION_KEYS: Final[frozenset[str]] = frozenset(
    {
        "ACCEPT", "ABSTAIN", "REJECT_INPUT", "OUT_OF_DISTRIBUTION",
        "CAPABILITY_UNAVAILABLE", "VALIDATION_REQUIRED",
    }
)


def _mapping(value: Any, name: str) -> dict[str, Any]:
    return bounded_json_mapping(value, name)


def _tree(value: Any, name: str) -> str:
    result = required_text(value, name, max_bytes=40)
    if not _TREE_RE.fullmatch(result):
        raise ResidualIntelligenceError(f"{name} must be an exact 40-character Git tree")
    return result


def _nonnegative_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ResidualIntelligenceError(f"{name} must be a non-negative integer")
    return value


def _require_mapping_keys(value: Mapping[str, Any], *, name: str, keys: frozenset[str]) -> None:
    missing = sorted(keys - set(value))
    if missing:
        raise ResidualIntelligenceError(f"{name} is missing required fields: {', '.join(missing)}")


@dataclass(frozen=True)
class ResidualGapReport:
    """Explicit blockers, nonclaims, and checks that were not performed."""

    blockers: tuple[str, ...]
    unsupported_claims: tuple[str, ...]
    not_run: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("blockers", "unsupported_claims", "not_run"):
            values = tuple(required_text(item, f"{name} item") for item in getattr(self, name))
            if len(values) != len(set(values)):
                raise ResidualIntelligenceError(f"{name} contains duplicate entries")
            object.__setattr__(self, name, values)

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "blockers": list(self.blockers),
            "unsupported_claims": list(self.unsupported_claims),
            "not_run": list(self.not_run),
        }


@dataclass(frozen=True)
class ResidualIntelligenceReleaseReport:
    """A bounded current-tree observation that cannot confer release authority."""

    start_tree: str
    end_tree: str
    corpus_admission_id: str
    expert_dispositions: Mapping[str, Any]
    before: Mapping[str, Any]
    after: Mapping[str, Any]
    costs: Mapping[str, Any]
    promotion_eligible: bool
    rollback_target: str
    gaps: ResidualGapReport
    lineage: Mapping[str, Any] = field(default_factory=dict)
    files_and_symbols: Mapping[str, Any] = field(default_factory=dict)
    corpus_rights_splits: Mapping[str, Any] = field(default_factory=dict)
    architecture_tokenizer_checkpoint: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, Any] = field(default_factory=dict)
    proof_validation: Mapping[str, Any] = field(default_factory=dict)
    drift: Mapping[str, Any] = field(default_factory=dict)
    promotion: Mapping[str, Any] = field(default_factory=dict)
    rollback: Mapping[str, Any] = field(default_factory=dict)
    report_authority: Mapping[str, Any] = field(default_factory=dict)
    schema: str = REPORT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != REPORT_SCHEMA:
            raise ResidualIntelligenceError("unsupported release report schema")
        object.__setattr__(self, "start_tree", _tree(self.start_tree, "start_tree"))
        object.__setattr__(self, "end_tree", _tree(self.end_tree, "end_tree"))
        if self.start_tree == self.end_tree:
            raise ResidualIntelligenceError("start_tree and end_tree must identify distinct snapshots")
        object.__setattr__(self, "corpus_admission_id", required_text(self.corpus_admission_id, "corpus_admission_id"))
        object.__setattr__(self, "rollback_target", required_text(self.rollback_target, "rollback_target"))
        if type(self.promotion_eligible) is not bool:
            raise ResidualIntelligenceError("promotion_eligible must be boolean")
        if self.promotion_eligible:
            raise ResidualIntelligenceError("release reports cannot promote")
        if not isinstance(self.gaps, ResidualGapReport):
            raise ResidualIntelligenceError("gaps must be a ResidualGapReport")
        for name in (
            "expert_dispositions", "before", "after", "costs", *_REQUIRED_SECTION_NAMES,
        ):
            object.__setattr__(self, name, _mapping(getattr(self, name), name))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema, "start_tree": self.start_tree, "end_tree": self.end_tree,
            "corpus_admission_id": self.corpus_admission_id,
            "expert_dispositions": dict(self.expert_dispositions), "before": dict(self.before),
            "after": dict(self.after), "costs": dict(self.costs), "promotion_eligible": False,
            "rollback_target": self.rollback_target, "gaps": self.gaps.to_dict(),
            "lineage": dict(self.lineage), "files_and_symbols": dict(self.files_and_symbols),
            "corpus_rights_splits": dict(self.corpus_rights_splits),
            "architecture_tokenizer_checkpoint": dict(self.architecture_tokenizer_checkpoint),
            "metrics": dict(self.metrics), "proof_validation": dict(self.proof_validation),
            "drift": dict(self.drift), "promotion": dict(self.promotion),
            "rollback": dict(self.rollback), "report_authority": dict(self.report_authority),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ResidualIntelligenceReleaseReport":
        if not isinstance(payload, Mapping):
            raise ResidualIntelligenceError("release report must be an object")
        fields = frozenset({
            "schema", "start_tree", "end_tree", "corpus_admission_id", "expert_dispositions",
            "before", "after", "costs", "promotion_eligible", "rollback_target", "gaps",
            *_REQUIRED_SECTION_NAMES,
        })
        unknown = sorted(str(key) for key in payload if key not in fields)
        missing = sorted(fields - set(payload))
        if unknown:
            raise ResidualIntelligenceError(f"release report contains unknown fields: {', '.join(unknown)}")
        if missing:
            raise ResidualIntelligenceError(f"release report is missing required fields: {', '.join(missing)}")
        gaps = payload.get("gaps")
        if not isinstance(gaps, Mapping) or set(gaps) != {"blockers", "unsupported_claims", "not_run"}:
            raise ResidualIntelligenceError("gaps must bind blockers, unsupported_claims, and not_run")
        return cls(
            schema=payload.get("schema"), start_tree=payload.get("start_tree"),
            end_tree=payload.get("end_tree"), corpus_admission_id=payload.get("corpus_admission_id"),
            expert_dispositions=payload.get("expert_dispositions"), before=payload.get("before"),
            after=payload.get("after"), costs=payload.get("costs"),
            promotion_eligible=payload.get("promotion_eligible"), rollback_target=payload.get("rollback_target"),
            gaps=ResidualGapReport(
                blockers=tuple(gaps["blockers"]), unsupported_claims=tuple(gaps["unsupported_claims"]),
                not_run=tuple(gaps["not_run"]),
            ),
            **{name: payload[name] for name in _REQUIRED_SECTION_NAMES},
        )


def _validate_metric_population(value: Mapping[str, Any], *, name: str) -> None:
    required = frozenset({
        "total_denominator", "evaluated", "ACCEPT", "ABSTAIN", "REJECT_INPUT",
        "OUT_OF_DISTRIBUTION", "CAPABILITY_UNAVAILABLE", "VALIDATION_REQUIRED",
        "true_accepts", "false_accepts",
    })
    _require_mapping_keys(value, name=name, keys=required)
    for key in required:
        _nonnegative_int(value[key], f"{name}.{key}")
    if value["evaluated"] > value["total_denominator"]:
        raise ResidualIntelligenceError(f"{name}.evaluated exceeds its denominator")
    if sum(value[key] for key in _DISPOSITION_KEYS) != value["evaluated"]:
        raise ResidualIntelligenceError(f"{name} dispositions do not preserve the evaluated denominator")
    if value["true_accepts"] + value["false_accepts"] != value["ACCEPT"]:
        raise ResidualIntelligenceError(f"{name} accept outcomes do not preserve their denominator")


def validate_release_claims(report: ResidualIntelligenceReleaseReport) -> ResidualIntelligenceReleaseReport:
    """Fail closed on incomplete, promotional, or denominator-losing reports."""

    if not isinstance(report, ResidualIntelligenceReleaseReport):
        raise ResidualIntelligenceError("validate_release_claims requires a typed release report")
    if report.promotion_eligible:
        raise ResidualIntelligenceError("release reports cannot promote")
    if set(report.gaps.unsupported_claims) != FORBIDDEN_CLAIMS:
        raise ResidualIntelligenceError("release report must enumerate every unsupported claim token")
    if not report.gaps.blockers or not report.gaps.not_run:
        raise ResidualIntelligenceError("release report must state blockers and not-run evidence")

    _require_mapping_keys(report.lineage, name="lineage", keys=frozenset({"start", "end", "snapshot_scope"}))
    start = _mapping(report.lineage["start"], "lineage.start")
    end = _mapping(report.lineage["end"], "lineage.end")
    _require_mapping_keys(start, name="lineage.start", keys=frozenset({"commit", "tree"}))
    _require_mapping_keys(end, name="lineage.end", keys=frozenset({"commit", "tree"}))
    if _tree(start["tree"], "lineage.start.tree") != report.start_tree:
        raise ResidualIntelligenceError("lineage start tree does not match start_tree")
    if _tree(end["tree"], "lineage.end.tree") != report.end_tree:
        raise ResidualIntelligenceError("lineage end tree does not match end_tree")
    _tree(start["commit"], "lineage.start.commit")
    _tree(end["commit"], "lineage.end.commit")
    required_text(report.lineage["snapshot_scope"], "lineage.snapshot_scope")

    _require_mapping_keys(report.files_and_symbols, name="files_and_symbols", keys=frozenset({"declared_outputs", "implementation_symbols", "snapshot_changed_files"}))
    for key in ("declared_outputs", "implementation_symbols", "snapshot_changed_files"):
        if not isinstance(report.files_and_symbols[key], list) or not report.files_and_symbols[key]:
            raise ResidualIntelligenceError(f"files_and_symbols.{key} must be a non-empty list")

    _require_mapping_keys(report.corpus_rights_splits, name="corpus_rights_splits", keys=frozenset({"admission_id", "decision", "rights_root", "split_root", "leakage_audit", "public_summary_only"}))
    if report.corpus_rights_splits["admission_id"] != report.corpus_admission_id:
        raise ResidualIntelligenceError("corpus admission identity does not match corpus_admission_id")
    if report.corpus_rights_splits["decision"] != "training_unavailable":
        raise ResidualIntelligenceError("report must not imply an admitted training corpus")
    if report.corpus_rights_splits["public_summary_only"] is not True:
        raise ResidualIntelligenceError("public release report must be bounded to summaries")

    _require_mapping_keys(report.architecture_tokenizer_checkpoint, name="architecture_tokenizer_checkpoint", keys=frozenset({"architecture", "tokenizer", "checkpoint"}))
    checkpoint = _mapping(report.architecture_tokenizer_checkpoint["checkpoint"], "checkpoint")
    if checkpoint.get("created") is not False or checkpoint.get("simulated") is not False:
        raise ResidualIntelligenceError("report cannot claim a real or simulated checkpoint")

    _require_mapping_keys(report.expert_dispositions, name="expert_dispositions", keys=frozenset({"registered_expert_count", "disposition_counts", "status"}))
    if _nonnegative_int(report.expert_dispositions["registered_expert_count"], "registered_expert_count") != 0:
        raise ResidualIntelligenceError("report must not invent a registered expert")
    dispositions = _mapping(report.expert_dispositions["disposition_counts"], "disposition_counts")
    if set(dispositions) != _DISPOSITION_KEYS:
        raise ResidualIntelligenceError("expert disposition counts must use the closed disposition set")
    if any(_nonnegative_int(value, f"disposition_counts.{key}") != 0 for key, value in dispositions.items()):
        raise ResidualIntelligenceError("report must not invent expert dispositions")

    _require_mapping_keys(report.metrics, name="metrics", keys=frozenset({"before", "after", "status"}))
    before = _mapping(report.metrics["before"], "metrics.before")
    after = _mapping(report.metrics["after"], "metrics.after")
    _validate_metric_population(before, name="metrics.before")
    _validate_metric_population(after, name="metrics.after")
    if before["total_denominator"] != after["total_denominator"]:
        raise ResidualIntelligenceError("before and after metrics must retain the same denominator")
    if dict(report.before) != before or dict(report.after) != after:
        raise ResidualIntelligenceError("legacy before/after projections must equal complete metric populations")

    _require_mapping_keys(report.costs, name="costs", keys=frozenset({"status", "denominators", "break_even"}))
    cost_denominators = _mapping(report.costs["denominators"], "costs.denominators")
    for key, value in cost_denominators.items():
        _nonnegative_int(value, f"costs.denominators.{key}")
    break_even = _mapping(report.costs["break_even"], "costs.break_even")
    if break_even.get("status") != "not_applicable":
        raise ResidualIntelligenceError("unqualified report must state break-even is not applicable")

    _require_mapping_keys(report.proof_validation, name="proof_validation", keys=frozenset({"proof_status", "validation_status", "evidence"}))
    _require_mapping_keys(report.drift, name="drift", keys=frozenset({"status", "routable", "required_action"}))
    if report.drift["routable"] is not False:
        raise ResidualIntelligenceError("unqualified experts cannot be reported as routable")
    _require_mapping_keys(report.promotion, name="promotion", keys=frozenset({"eligible", "decision", "hard_gate_status"}))
    if report.promotion["eligible"] is not False or report.promotion["decision"] != "not_eligible":
        raise ResidualIntelligenceError("report must preserve its non-promotional disposition")
    _require_mapping_keys(report.rollback, name="rollback", keys=frozenset({"target", "report_rebuild_tree", "policy"}))
    if report.rollback["target"] != report.rollback_target:
        raise ResidualIntelligenceError("rollback target does not match rollback section")
    if _tree(report.rollback["report_rebuild_tree"], "rollback.report_rebuild_tree") != report.end_tree:
        raise ResidualIntelligenceError("report rollback must rebuild from the inspected end tree")
    _require_mapping_keys(report.report_authority, name="report_authority", keys=frozenset({"completion_authoritative", "promotion_authoritative", "proof_authoritative"}))
    if any(report.report_authority[key] is not False for key in report.report_authority):
        raise ResidualIntelligenceError("release report cannot claim any authority")
    return report


__all__ = (
    "FORBIDDEN_CLAIMS", "REPORT_SCHEMA", "ResidualGapReport",
    "ResidualIntelligenceReleaseReport", "validate_release_claims",
)
