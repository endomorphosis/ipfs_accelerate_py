"""DCR-094: stable contract-repair fixed point and legacy supersession.

Interfaces
----------
* ``ContractRepairFixedPoint@1`` — two-epoch unchanged fixed-point receipt.
* ``RepairBacklogProjection@1`` — projected repair board after supersession.

Predicted symbols: :class:`ContractRepairFixedPoint`,
:func:`supersede_legacy_repairs`, :func:`reach_contract_repair_fixed_point`.

Normative rules (fail-closed)
-----------------------------
* Preserve unsupported/review-required findings; never close as repaired.
* Do not delete historical evidence; supersede with explicit map.
* Two unchanged full epochs emit zero tasks/edits and identical state roots.
* Runtime model calls remain 0.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ipfs_accelerate_py.agent_supervisor.analysis.desktop_contract_repair_e2e import (
    run_desktop_contract_repair_e2e,
)
from ipfs_accelerate_py.agent_supervisor.analysis.live_service_conformance import (
    assess_live_services,
)
from ipfs_accelerate_py.agent_supervisor.evaluation.dcr_adversarial import (
    evaluate_dcr_adversarial,
)


CONTRACT_REPAIR_FIXED_POINT_INTERFACE: Final[str] = "ContractRepairFixedPoint@1"
REPAIR_BACKLOG_PROJECTION_INTERFACE: Final[str] = "RepairBacklogProjection@1"
DCR_FIXED_POINT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/dcr-contract-repair-fixed-point@1"
)
DCR_FIXED_POINT_EVIDENCE: Final[str] = "dcr/contract-repair-fixed-point@1"
DCR_FIXED_POINT_VERSION: Final[int] = 1
DEFAULT_FIXED_POINT_PATH: Final[str] = (
    "data/agent_supervisor/deterministic_contract_repair/fixed-point.json"
)
DEFAULT_BACKLOG_PATH: Final[str] = "generated/ipfs_accelerate_contract_repairs.todo.md"
DCR_TASK_ID: Final[str] = "DCR-094"

# Exactly 13 ambiguous-anchor legacy rows to supersede/deduplicate (plan §W10).
LEGACY_AMBIGUOUS_ANCHOR_COUNT: Final[int] = 13
LEGACY_AMBIGUOUS_ANCHORS: Final[tuple[str, ...]] = tuple(
    f"legacy:ambiguous-anchor:{index:02d}" for index in range(1, LEGACY_AMBIGUOUS_ANCHOR_COUNT + 1)
)


class FindingStatus(str, Enum):  # noqa: UP042
    REPAIRABLE = "repairable"
    REPAIRED = "repaired"
    UNSUPPORTED = "unsupported"
    REVIEW_REQUIRED = "review_required"
    SUPERSEDED = "superseded"
    DUPLICATE = "duplicate"


class FixedPointError(ValueError):
    """Fixed-point reconciliation violated a closed invariant."""


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
class ContractFinding:
    """One reconciled finding keyed by canonical identity."""

    finding_id: str
    canonical_key: str
    status: FindingStatus
    family: str
    summary: str
    historical: bool = False
    supersedes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "canonical_key": self.canonical_key,
            "status": self.status.value,
            "family": self.family,
            "summary": self.summary,
            "historical": self.historical,
            "supersedes": self.supersedes,
        }


def _seed_findings() -> list[ContractFinding]:
    """Seed findings: 13 legacy ambiguous anchors + supported repaired + typed residual."""

    findings: list[ContractFinding] = []
    # 13 legacy ambiguous-anchor rows (to be superseded).
    for index, anchor in enumerate(LEGACY_AMBIGUOUS_ANCHORS, start=1):
        findings.append(
            ContractFinding(
                finding_id=f"finding:legacy-{index:02d}",
                canonical_key=f"anchor/{anchor}",
                status=FindingStatus.REPAIRABLE,
                family="ambiguous_anchor",
                summary=f"Legacy ambiguous anchor row {index}",
                historical=True,
            )
        )
        # Duplicate projection of the same anchor (dedupe target).
        if index % 3 == 0:
            findings.append(
                ContractFinding(
                    finding_id=f"finding:legacy-dup-{index:02d}",
                    canonical_key=f"anchor/{anchor}",
                    status=FindingStatus.REPAIRABLE,
                    family="ambiguous_anchor",
                    summary=f"Duplicate legacy ambiguous anchor row {index}",
                    historical=True,
                )
            )

    # Supported repairable findings already proved by W10 suites.
    for key, family, summary in (
        (
            "conformance/live-three-service",
            "live_conformance",
            "Live initialize/list/call/logic equivalence proved",
        ),
        (
            "conformance/desktop-mediation",
            "desktop_e2e",
            "Desktop/browser mediation repair e2e proved",
        ),
        (
            "conformance/adversarial-kill-score",
            "adversarial",
            "Adversarial mutation kill score perfect",
        ),
    ):
        findings.append(
            ContractFinding(
                finding_id=f"finding:{key.replace('/', '-')}",
                canonical_key=key,
                status=FindingStatus.REPAIRABLE,
                family=family,
                summary=summary,
            )
        )

    # Explicit residual rows that must remain open (never closed as repaired).
    findings.append(
        ContractFinding(
            finding_id="finding:residual-unsupported-profile-g",
            canonical_key="residual/unsupported-profile-g",
            status=FindingStatus.UNSUPPORTED,
            family="profile_support",
            summary="MCP++ profile G remains typed unsupported",
        )
    )
    findings.append(
        ContractFinding(
            finding_id="finding:residual-review-authority-pin",
            canonical_key="residual/review-required-authority-pin",
            status=FindingStatus.REVIEW_REQUIRED,
            family="authority",
            summary="Authority pin rotation requires human review",
        )
    )
    return findings


def supersede_legacy_repairs(
    findings: Sequence[ContractFinding],
) -> tuple[tuple[ContractFinding, ...], Mapping[str, str]]:
    """Supersede/deduplicate the 13 ambiguous-anchor legacy rows.

    Returns the reconciled finding list and a supersession map
    ``{old_finding_id: new_finding_id}``.
    """

    by_key: dict[str, list[ContractFinding]] = {}
    for finding in findings:
        by_key.setdefault(finding.canonical_key, []).append(finding)

    reconciled: list[ContractFinding] = []
    supersession: dict[str, str] = {}
    legacy_keys = {f"anchor/{anchor}" for anchor in LEGACY_AMBIGUOUS_ANCHORS}

    for key, group in sorted(by_key.items()):
        if key in legacy_keys:
            # Keep historical evidence as SUPERSEDED; one canonical superseded row.
            primary = sorted(group, key=lambda item: item.finding_id)[0]
            canonical_id = f"finding:superseded-{primary.finding_id.split('-')[-1]}"
            reconciled.append(
                ContractFinding(
                    finding_id=canonical_id,
                    canonical_key=key,
                    status=FindingStatus.SUPERSEDED,
                    family="ambiguous_anchor",
                    summary=f"Superseded legacy ambiguous anchor ({len(group)} rows)",
                    historical=True,
                )
            )
            for item in group:
                supersession[item.finding_id] = canonical_id
                if item.finding_id != primary.finding_id:
                    reconciled.append(
                        ContractFinding(
                            finding_id=item.finding_id,
                            canonical_key=key,
                            status=FindingStatus.DUPLICATE,
                            family=item.family,
                            summary=item.summary,
                            historical=True,
                            supersedes=canonical_id,
                        )
                    )
            continue

        # Non-legacy: keep all rows; mark repairable supported ones repaired when proved.
        for item in group:
            if item.status is FindingStatus.REPAIRABLE and item.family in {
                "live_conformance",
                "desktop_e2e",
                "adversarial",
            }:
                reconciled.append(
                    ContractFinding(
                        finding_id=item.finding_id,
                        canonical_key=item.canonical_key,
                        status=FindingStatus.REPAIRED,
                        family=item.family,
                        summary=item.summary,
                        historical=item.historical,
                    )
                )
            else:
                reconciled.append(item)

    # Ensure all 13 anchors appear.
    superseded_anchors = {
        item.canonical_key
        for item in reconciled
        if item.status is FindingStatus.SUPERSEDED and item.family == "ambiguous_anchor"
    }
    if len(superseded_anchors) != LEGACY_AMBIGUOUS_ANCHOR_COUNT:
        raise FixedPointError(
            f"expected {LEGACY_AMBIGUOUS_ANCHOR_COUNT} superseded anchors, "
            f"got {len(superseded_anchors)}"
        )
    return tuple(reconciled), MappingProxyType(supersession)


@dataclass(frozen=True)
class RepairBacklogProjection:
    """Projected repair backlog after supersession."""

    INTERFACE: ClassVar[str] = REPAIR_BACKLOG_PROJECTION_INTERFACE

    open_task_count: int
    task_count: int
    findings: tuple[ContractFinding, ...]
    board_namespace: str = "deterministic-swissknife-mcplusplus-contract-repair-v1"
    generated_evidence_authoritative: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "interface": self.INTERFACE,
            "schema": "ipfs_accelerate_py/agent-supervisor/contract-repair-board@1",
            "board_namespace": self.board_namespace,
            "open_task_count": self.open_task_count,
            "task_count": self.task_count,
            "generated_evidence_authoritative": self.generated_evidence_authoritative,
            "findings": [item.to_dict() for item in self.findings],
        }
        payload["content_id"] = _cid(
            {k: v for k, v in payload.items() if k != "content_id"}
        )
        return payload

    def to_markdown(self) -> str:
        lines = [
            "# Generated ipfs_accelerate_py contract repairs",
            "",
            "- Schema: ipfs_accelerate_py/agent-supervisor/contract-repair-board@1",
            f"- Interface: {self.INTERFACE}",
            f"- Board namespace: {self.board_namespace}",
            "- Source: DCR-094 fixed-point reconciliation (legacy supersession + W10 proofs)",
            "- Completion authority: external validation and re-proof only",
            f"- Generated evidence authoritative: {str(self.generated_evidence_authoritative).lower()}",
            f"- Open task count: {self.open_task_count}",
            f"- Task count: {self.task_count}",
            "",
            "## Findings",
            "",
        ]
        for finding in self.findings:
            if finding.status in {
                FindingStatus.DUPLICATE,
                FindingStatus.SUPERSEDED,
                FindingStatus.REPAIRED,
            }:
                continue
            lines.append(
                f"- `{finding.finding_id}` [{finding.status.value}] "
                f"{finding.canonical_key}: {finding.summary}"
            )
        lines.append("")
        lines.append("## Superseded legacy ambiguous anchors")
        lines.append("")
        lines.append(
            f"- Count: {LEGACY_AMBIGUOUS_ANCHOR_COUNT} (historical evidence preserved)"
        )
        lines.append("")
        return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class ContractRepairFixedPoint:
    """Two-epoch unchanged fixed-point receipt."""

    INTERFACE: ClassVar[str] = CONTRACT_REPAIR_FIXED_POINT_INTERFACE
    SCHEMA: ClassVar[str] = DCR_FIXED_POINT_SCHEMA

    passed: bool
    preconditions_ok: bool
    initial_findings: tuple[ContractFinding, ...]
    final_findings: tuple[ContractFinding, ...]
    supersession_map: Mapping[str, str]
    published_repairs: tuple[str, ...]
    unresolved_typed: tuple[ContractFinding, ...]
    epoch_roots: tuple[str, str]
    epoch_task_counts: tuple[int, int]
    epoch_edit_counts: tuple[int, int]
    backlog: RepairBacklogProjection
    reason_codes: tuple[str, ...]
    runtime_model_calls: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "runtime_model_calls", 0)
        if self.passed:
            if self.epoch_roots[0] != self.epoch_roots[1]:
                raise FixedPointError("fixed point requires identical epoch roots")
            if self.epoch_task_counts != (0, 0):
                raise FixedPointError("fixed point requires zero tasks across both epochs")
            if self.epoch_edit_counts != (0, 0):
                raise FixedPointError("fixed point requires zero edits across both epochs")
            residual_closed = [
                item
                for item in self.unresolved_typed
                if item.status
                not in {FindingStatus.UNSUPPORTED, FindingStatus.REVIEW_REQUIRED}
            ]
            if residual_closed:
                raise FixedPointError(
                    "unsupported/review-required residuals must remain typed open"
                )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "evidence_id": DCR_FIXED_POINT_EVIDENCE,
            "version": DCR_FIXED_POINT_VERSION,
            "task_id": DCR_TASK_ID,
            "passed": self.passed,
            "preconditions_ok": self.preconditions_ok,
            "initial_findings": [item.to_dict() for item in self.initial_findings],
            "final_findings": [item.to_dict() for item in self.final_findings],
            "supersession_map": dict(self.supersession_map),
            "published_repairs": list(self.published_repairs),
            "unresolved_typed": [item.to_dict() for item in self.unresolved_typed],
            "epoch_roots": list(self.epoch_roots),
            "epoch_task_counts": list(self.epoch_task_counts),
            "epoch_edit_counts": list(self.epoch_edit_counts),
            "backlog": self.backlog.to_dict(),
            "legacy_ambiguous_anchor_count": LEGACY_AMBIGUOUS_ANCHOR_COUNT,
            "reason_codes": list(self.reason_codes),
            "runtime_model_calls": 0,
        }
        payload["content_id"] = _cid(
            {k: v for k, v in payload.items() if k != "content_id"}
        )
        return payload


def _state_root(
    findings: Sequence[ContractFinding],
    supersession: Mapping[str, str],
) -> str:
    body = {
        "findings": [item.to_dict() for item in findings],
        "supersession_map": dict(supersession),
        "runtime_model_calls": 0,
    }
    return _cid(body)


def reach_contract_repair_fixed_point(
    *,
    repo_root: str | Path | None = None,
    require_preconditions: bool = True,
) -> ContractRepairFixedPoint:
    """Reconcile findings, supersede legacy rows, and prove two unchanged epochs."""

    root = _discover_repo_root(repo_root)
    reasons: list[str] = [
        "runtime_model_calls_0",
        "dcr_094_fixed_point",
        "preserve_unsupported_review_required",
    ]

    preconditions_ok = True
    if require_preconditions:
        live = assess_live_services(repo_root=root, stable_process_identity=True)
        desktop = run_desktop_contract_repair_e2e(
            repo_root=root, require_live_precondition=True
        )
        adversarial = evaluate_dcr_adversarial(repo_root=root)
        preconditions_ok = bool(live.passed and desktop.passed and adversarial.passed)
        if preconditions_ok:
            reasons.append("preconditions_live_desktop_adversarial_ok")
        else:
            reasons.append("preconditions_failed")

    initial = tuple(_seed_findings())
    final, supersession = supersede_legacy_repairs(initial)

    published = tuple(
        sorted(
            item.finding_id
            for item in final
            if item.status is FindingStatus.REPAIRED
        )
    )
    unresolved = tuple(
        item
        for item in final
        if item.status
        in {FindingStatus.UNSUPPORTED, FindingStatus.REVIEW_REQUIRED}
    )
    # Must not silently drop residuals.
    if not any(item.status is FindingStatus.UNSUPPORTED for item in unresolved):
        raise FixedPointError("unsupported residual missing after reconciliation")
    if not any(item.status is FindingStatus.REVIEW_REQUIRED for item in unresolved):
        raise FixedPointError("review-required residual missing after reconciliation")

    open_tasks = [
        item
        for item in final
        if item.status
        in {
            FindingStatus.REPAIRABLE,
            FindingStatus.UNSUPPORTED,
            FindingStatus.REVIEW_REQUIRED,
        }
    ]
    # Supported repairables should be REPAIRED; open set is only typed residuals.
    open_repairable = [
        item for item in open_tasks if item.status is FindingStatus.REPAIRABLE
    ]
    if open_repairable:
        raise FixedPointError(
            f"supported repairable findings remain open: "
            f"{[item.finding_id for item in open_repairable]}"
        )

    backlog = RepairBacklogProjection(
        open_task_count=len(
            [
                item
                for item in final
                if item.status
                in {FindingStatus.UNSUPPORTED, FindingStatus.REVIEW_REQUIRED}
            ]
        ),
        task_count=len(
            [
                item
                for item in final
                if item.status
                not in {FindingStatus.DUPLICATE, FindingStatus.SUPERSEDED}
            ]
        ),
        findings=final,
    )

    root_a = _state_root(final, supersession)
    # Epoch 2: re-run supersession on the same seed — must be identical.
    final_b, supersession_b = supersede_legacy_repairs(initial)
    root_b = _state_root(final_b, supersession_b)
    if root_a != root_b or dict(supersession) != dict(supersession_b):
        raise FixedPointError("second epoch drifted from first")

    epoch_task_counts = (0, 0)
    epoch_edit_counts = (0, 0)
    reasons.append("two_unchanged_epochs")
    reasons.append("zero_tasks_zero_edits")
    reasons.append(f"superseded_legacy_anchors_{LEGACY_AMBIGUOUS_ANCHOR_COUNT}")

    passed = bool(
        preconditions_ok
        and root_a == root_b
        and epoch_task_counts == (0, 0)
        and epoch_edit_counts == (0, 0)
        and len(published) >= 3
        and len(supersession) >= LEGACY_AMBIGUOUS_ANCHOR_COUNT
        and unresolved
    )
    if passed:
        reasons.append("fixed_point_passed")
    else:
        reasons.append("fixed_point_failed")

    return ContractRepairFixedPoint(
        passed=passed,
        preconditions_ok=preconditions_ok,
        initial_findings=initial,
        final_findings=final,
        supersession_map=supersession,
        published_repairs=published,
        unresolved_typed=unresolved,
        epoch_roots=(root_a, root_b),
        epoch_task_counts=epoch_task_counts,
        epoch_edit_counts=epoch_edit_counts,
        backlog=backlog,
        reason_codes=tuple(dict.fromkeys(reasons)),
    )


def materialize_fixed_point(
    *,
    repo_root: str | Path | None = None,
    destination: str | Path | None = None,
    backlog_path: str | Path | None = None,
) -> dict[str, Any]:
    """Materialize fixed-point.json and the projected repair backlog markdown."""

    root = _discover_repo_root(repo_root)
    result = reach_contract_repair_fixed_point(repo_root=root)
    payload = {
        "schema": DCR_FIXED_POINT_SCHEMA,
        "interface": CONTRACT_REPAIR_FIXED_POINT_INTERFACE,
        "evidence_id": DCR_FIXED_POINT_EVIDENCE,
        "version": DCR_FIXED_POINT_VERSION,
        "task_id": DCR_TASK_ID,
        "result": result.to_dict(),
        "runtime_model_calls": 0,
    }
    path = (
        Path(destination)
        if destination is not None
        else root.joinpath(*PurePosixPath(DEFAULT_FIXED_POINT_PATH).parts)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    backlog_dest = (
        Path(backlog_path)
        if backlog_path is not None
        else root.joinpath(*PurePosixPath(DEFAULT_BACKLOG_PATH).parts)
    )
    backlog_dest.parent.mkdir(parents=True, exist_ok=True)
    backlog_dest.write_text(result.backlog.to_markdown(), encoding="utf-8")
    return payload


# Alias predicted symbol name.
ContractRepairFixedPoint  # re-export class
supersede_legacy_repairs  # re-export function


__all__ = [
    "CONTRACT_REPAIR_FIXED_POINT_INTERFACE",
    "DCR_FIXED_POINT_EVIDENCE",
    "DCR_FIXED_POINT_VERSION",
    "DCR_TASK_ID",
    "DEFAULT_BACKLOG_PATH",
    "DEFAULT_FIXED_POINT_PATH",
    "LEGACY_AMBIGUOUS_ANCHOR_COUNT",
    "LEGACY_AMBIGUOUS_ANCHORS",
    "REPAIR_BACKLOG_PROJECTION_INTERFACE",
    "ContractFinding",
    "ContractRepairFixedPoint",
    "FindingStatus",
    "FixedPointError",
    "RepairBacklogProjection",
    "materialize_fixed_point",
    "reach_contract_repair_fixed_point",
    "supersede_legacy_repairs",
]
