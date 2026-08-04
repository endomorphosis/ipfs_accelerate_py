"""Automatic progress recovery for stalled implementation work.

Wave-0 production operation repeatedly stalled on a small set of recoverable
conditions:

* declared task outputs already present on the merge target (operator land or
  prior successful write) while the daemon kept re-implementing and burning
  repair-round budget
* identical repair-round budget exhaustion after transient provider/context
  failures even after the repository tree advanced
* dead-owner worktree lifecycle claims that hold unexpired leases for hours

This module keeps the recovery policy pure and unit-testable.  The daemon
applies the resulting decisions against durable state and lifecycle stores.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

# Failure markers that mean "retrying the same implement path will not help
# until state is recovered" once declared outputs are already on the merge
# target.
PROGRESS_RECOVERY_FAILURE_MARKERS: Final[tuple[str, ...]] = (
    "implementation_repair_round_budget_exhausted",
    "identical_implementation_failure_escalated",
    "identical_implementation_failure_backoff",
    "production source context is unavailable or insufficient",
    "production_source_context",
    "worktree_lifecycle_claim_exists",
    "owner_dead_lease_unexpired",
    "lifecycle_race",
    "provider_review_pending",
    "no_change_implementation_binding_recovery_failed",
    "prior_merged_implementation_binding_missing",
)


@dataclass(frozen=True)
class DeclaredOutputPresence:
    """Whether a task's declared outputs are fully present on a tree root."""

    task_id: str
    declared: tuple[str, ...]
    present: tuple[str, ...]
    missing: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return bool(self.declared) and not self.missing


@dataclass(frozen=True)
class ProgressRecoveryDecision:
    """One automatic recovery action for a stalled implementation task."""

    task_id: str
    action: str
    reason: str
    reset_attempt_budget: bool = False
    clear_diagnostics: bool = False
    reclaim_dead_lifecycle: bool = False
    treat_as_landed_outputs: bool = False
    details: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "task_id": self.task_id,
            "action": self.action,
            "reason": self.reason,
            "reset_attempt_budget": self.reset_attempt_budget,
            "clear_diagnostics": self.clear_diagnostics,
            "reclaim_dead_lifecycle": self.reclaim_dead_lifecycle,
            "treat_as_landed_outputs": self.treat_as_landed_outputs,
        }
        if self.details:
            payload["details"] = dict(self.details)
        return payload


def declared_output_presence(
    *,
    task_id: str,
    outputs: Sequence[str],
    repo_root: Path,
) -> DeclaredOutputPresence:
    """Report which declared outputs already exist under ``repo_root``."""

    declared = tuple(
        str(path).strip().replace("\\", "/")
        for path in outputs
        if str(path).strip()
    )
    present: list[str] = []
    missing: list[str] = []
    root = Path(repo_root)
    for path in declared:
        candidate = root / path
        if candidate.is_file():
            present.append(path)
        else:
            missing.append(path)
    return DeclaredOutputPresence(
        task_id=str(task_id),
        declared=declared,
        present=tuple(present),
        missing=tuple(missing),
    )


def _text_matches_recovery_markers(text: str) -> bool:
    lowered = str(text or "").casefold()
    if not lowered:
        return False
    return any(marker.casefold() in lowered for marker in PROGRESS_RECOVERY_FAILURE_MARKERS)


def should_recover_stalled_task(
    *,
    task_id: str,
    outputs: Sequence[str],
    repo_root: Path,
    attempt_count: int,
    max_repair_rounds: int,
    last_returncode: int | None,
    last_failure_text: str = "",
    selection_idle_reason: str = "",
    implementation_in_progress: bool = False,
    active_task_id: str = "",
) -> ProgressRecoveryDecision | None:
    """Decide whether a stalled task should be auto-recovered.

    Recovery is conservative: only when *all* declared outputs are already on
    the merge-target tree, so re-implementation would thrash without adding
    product value.
    """

    if implementation_in_progress and str(active_task_id or "") == str(task_id):
        return None

    presence = declared_output_presence(
        task_id=task_id,
        outputs=outputs,
        repo_root=repo_root,
    )
    if not presence.complete:
        return None

    attempt_count = max(0, int(attempt_count or 0))
    max_repair_rounds = max(1, int(max_repair_rounds or 1))
    repair_budget_exhausted = attempt_count - 1 > max_repair_rounds or attempt_count > max_repair_rounds
    failure_marker = _text_matches_recovery_markers(
        f"{last_failure_text}\n{selection_idle_reason}"
    )
    failed_last = last_returncode not in (None, 0)

    if not (repair_budget_exhausted or failure_marker or failed_last or attempt_count > 0):
        # Quiet recognition only — the production landed-task guard short-circuits
        # re-implementation from product presence without rewriting durable state.
        return ProgressRecoveryDecision(
            task_id=str(task_id),
            action="recognize_landed_outputs",
            reason="declared_outputs_present_on_merge_target",
            reset_attempt_budget=False,
            clear_diagnostics=False,
            reclaim_dead_lifecycle=False,
            treat_as_landed_outputs=True,
            details={
                "present": list(presence.present),
                "attempt_count": attempt_count,
            },
        )

    return ProgressRecoveryDecision(
        task_id=str(task_id),
        action="reset_stalled_landed_task",
        reason=(
            "repair_budget_or_failure_with_landed_outputs"
            if repair_budget_exhausted or failure_marker
            else "failed_attempt_with_landed_outputs"
        ),
        reset_attempt_budget=True,
        clear_diagnostics=True,
        reclaim_dead_lifecycle=True,
        treat_as_landed_outputs=True,
        details={
            "present": list(presence.present),
            "attempt_count": attempt_count,
            "max_repair_rounds": max_repair_rounds,
            "last_returncode": last_returncode,
            "selection_idle_reason": str(selection_idle_reason or ""),
            "failure_marker": failure_marker,
        },
    )


def operator_landed_binding_payload(
    *,
    task_id: str,
    canonical_task_cid: str,
    merge_commit: str,
    repository_tree_id: str,
    present_outputs: Sequence[str],
) -> dict[str, Any]:
    """Build a recovered-binding payload for outputs already on the merge target.

    This is intentionally weaker than a merge-train receipt: it proves product
    presence on the exact merge-target tree, not independent provider review.
    Callers must keep completion non-authoritative.
    """

    commit = str(merge_commit or "").strip()
    tree_id = str(repository_tree_id or "").strip()
    return {
        "recovered": True,
        "reason": "declared_outputs_present_on_merge_target",
        "reason_codes": ["declared_outputs_present_on_merge_target"],
        "task_id": str(task_id),
        "canonical_task_cid": str(canonical_task_cid),
        "implementation_commit": commit,
        "prior_merge_commit": commit,
        "prior_repository_tree_id": tree_id,
        "merge_commit": commit,
        "repository_tree_id": tree_id,
        "validation_result": {},
        "gate_evidence": {},
        "model_invocation_observed": False,
        "source": "merge_target_declared_outputs",
        "source_id": f"outputs:{','.join(sorted(str(p) for p in present_outputs))}",
        "present_outputs": list(present_outputs),
        "completion_authoritative": False,
        "proof_authoritative": False,
    }


__all__ = [
    "PROGRESS_RECOVERY_FAILURE_MARKERS",
    "DeclaredOutputPresence",
    "ProgressRecoveryDecision",
    "declared_output_presence",
    "operator_landed_binding_payload",
    "should_recover_stalled_task",
]
