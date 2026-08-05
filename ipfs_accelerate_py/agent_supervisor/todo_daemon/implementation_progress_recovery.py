"""Automatic progress recovery for stalled implementation work.

Wave-0 production operation repeatedly stalled on a small set of recoverable
conditions:

* declared task outputs already present on the merge target (operator land or
  prior successful write) while the daemon kept re-implementing and burning
  repair-round budget
* identical repair-round budget exhaustion after transient provider/context
  failures even after the repository tree advanced
* open board work with missing products permanently fenced by
  ``implementation_repair_round_budget_exhausted`` (ASE2-007) after transient
  ProviderRoutingError / review-capacity failures
* dead-owner worktree lifecycle claims that hold unexpired leases for hours
* independent-review-pending thrash: landed guard fires, integration fails
  closed, recovery resets attempts, and the task is re-selected forever
* board status left ``todo`` after products land so dependents never unlock

This module keeps the recovery policy pure and unit-testable.  The daemon
applies the resulting decisions against durable state and lifecycle stores.

Completions produced here are intentionally non-authoritative: product
presence proves implement work finished, not dual-review acceptance.
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
    # Thrash / landed-integration paths observed in wave0.
    "implementation_not_integrated",
    "landed_products_provider_review_pending",
    "landed_binding_has_no_typed_provider_receipt",
    "provider_review_pending_no_reimplementation",
    "legacy_landed_review",
    "landed_task_guard",
)

# Subset that means products are already present and only review/bookkeeping
# remains — never re-open attempt budget for a fresh implement loop.
LANDED_REVIEW_PENDING_MARKERS: Final[tuple[str, ...]] = (
    "provider_review_pending",
    "landed_products_provider_review_pending",
    "landed_binding_has_no_typed_provider_receipt",
    "provider_review_pending_no_reimplementation",
    "implementation_not_integrated",
    "legacy_landed_review",
    "landed_task_guard",
)

LIFECYCLE_RECLAIM_MARKERS: Final[tuple[str, ...]] = (
    "worktree_lifecycle_claim_exists",
    "owner_dead_lease_unexpired",
    "lifecycle_race",
)

# Open-board work with missing products: these markers mean the implement
# loop can no longer select the task until attempt/diagnostic state is reset.
OPEN_WORK_BUDGET_EXHAUSTION_MARKERS: Final[tuple[str, ...]] = (
    "implementation_repair_round_budget_exhausted",
    "identical_implementation_failure_escalated",
    "identical_implementation_failure_backoff",
    "production source context is unavailable or insufficient",
    "production_source_context",
    "worktree_lifecycle_claim_exists",
    "owner_dead_lease_unexpired",
    "lifecycle_race",
    "providerroutingerror",
    "codex_quota_exhausted",
)

# Park review-pending landed work for an hour so lanes move to real backlog.
DEFAULT_LANDED_REVIEW_DEFER_SECONDS: Final[int] = 3_600

# Brief pause after open-work budget reset so a tight ProviderRoutingError
# loop cannot burn the fresh budget in a single second.
DEFAULT_OPEN_WORK_RESET_COOLDOWN_SECONDS: Final[int] = 60


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
    defer_review_pending: bool = False
    defer_seconds: int = 0
    soft_complete_board: bool = False
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
            "defer_review_pending": self.defer_review_pending,
            "defer_seconds": int(self.defer_seconds or 0),
            "soft_complete_board": self.soft_complete_board,
            "completion_authoritative": False,
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

    declared = tuple(str(path).strip().replace("\\", "/") for path in outputs if str(path).strip())
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


def _text_matches_markers(text: str, markers: Sequence[str]) -> bool:
    lowered = str(text or "").casefold()
    if not lowered:
        return False
    return any(marker.casefold() in lowered for marker in markers)


def _text_matches_recovery_markers(text: str) -> bool:
    return _text_matches_markers(text, PROGRESS_RECOVERY_FAILURE_MARKERS)


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
    board_status: str = "",
) -> ProgressRecoveryDecision | None:
    """Decide whether a stalled task should be auto-recovered.

    Two complementary policies:

    * **Landed products** (all declared outputs on the merge-target tree): park
      for independent review / soft-close the board. Never re-open attempt
      budget — that re-selected landed tasks forever.

    * **Open work with missing products**: when the implement loop has burned
      its repair-round budget (or is deferred for that reason) while the board
      still lists the task as open, reset attempt budget and clear diagnostics
      so residual wave work can relaunch. Without this path ASE2-007 remained
      permanently fenced after transient ProviderRoutingError / review-capacity
      failures even though no product existed on tip.
    """

    if implementation_in_progress and str(active_task_id or "") == str(task_id):
        return None

    presence = declared_output_presence(
        task_id=task_id,
        outputs=outputs,
        repo_root=repo_root,
    )
    attempt_count = max(0, int(attempt_count or 0))
    max_repair_rounds = max(1, int(max_repair_rounds or 1))
    combined_text = f"{last_failure_text}\n{selection_idle_reason}"
    repair_budget_exhausted = (
        attempt_count - 1 > max_repair_rounds or attempt_count > max_repair_rounds
    )
    failure_marker = _text_matches_recovery_markers(combined_text)
    review_pending_marker = _text_matches_markers(combined_text, LANDED_REVIEW_PENDING_MARKERS)
    lifecycle_marker = _text_matches_markers(combined_text, LIFECYCLE_RECLAIM_MARKERS)
    open_work_budget_marker = _text_matches_markers(
        combined_text, OPEN_WORK_BUDGET_EXHAUSTION_MARKERS
    )
    failed_last = last_returncode not in (None, 0)
    normalized_board = str(board_status or "").strip().lower()
    board_already_completed = normalized_board in {
        "completed",
        "done",
        "complete",
    }
    board_open = normalized_board not in {
        "completed",
        "done",
        "complete",
        "blocked",
        "on_hold",
        "cancelled",
        "canceled",
    }

    details: dict[str, Any] = {
        "present": list(presence.present),
        "missing": list(presence.missing),
        "attempt_count": attempt_count,
        "max_repair_rounds": max_repair_rounds,
        "last_returncode": last_returncode,
        "selection_idle_reason": str(selection_idle_reason or ""),
        "failure_marker": failure_marker,
        "review_pending_marker": review_pending_marker,
        "lifecycle_marker": lifecycle_marker,
        "open_work_budget_marker": open_work_budget_marker,
        "board_status": str(board_status or ""),
    }

    if not presence.complete:
        # Products still missing: re-open a burned attempt budget so residual
        # ready board work is not permanently fenced. Require an explicit
        # exhaustion signal so ordinary in-flight tasks are untouched.
        open_work_exhausted = repair_budget_exhausted or open_work_budget_marker
        if open_work_exhausted and board_open and bool(presence.declared):
            return ProgressRecoveryDecision(
                task_id=str(task_id),
                action="reset_open_work_attempt_budget",
                reason="repair_budget_exhausted_with_missing_outputs",
                reset_attempt_budget=True,
                clear_diagnostics=True,
                reclaim_dead_lifecycle=True,
                treat_as_landed_outputs=False,
                # Do not use defer_review_pending: the apply path floors that
                # path at DEFAULT_LANDED_REVIEW_DEFER_SECONDS (1h) and would
                # re-fence residual open work. Cooldown is applied separately
                # via local retry-not-before when the daemon applies the reset.
                defer_review_pending=False,
                defer_seconds=DEFAULT_OPEN_WORK_RESET_COOLDOWN_SECONDS,
                soft_complete_board=False,
                details=details,
            )
        return None

    # Landed products + any thrash / failure / repair burn → park for review.
    # Do not re-open attempt budget: that re-selects implement forever.
    landed_thrash = (
        review_pending_marker
        or failed_last
        or repair_budget_exhausted
        or failure_marker
        or attempt_count > 0
    )
    if landed_thrash:
        return ProgressRecoveryDecision(
            task_id=str(task_id),
            action="defer_landed_review_pending",
            reason=(
                "landed_outputs_review_pending"
                if review_pending_marker or not failure_marker
                else "repair_budget_or_failure_with_landed_outputs"
            ),
            # Critical: do not reset attempt budget — that re-enables thrash.
            reset_attempt_budget=False,
            clear_diagnostics=True,
            reclaim_dead_lifecycle=True,
            treat_as_landed_outputs=True,
            defer_review_pending=True,
            defer_seconds=DEFAULT_LANDED_REVIEW_DEFER_SECONDS,
            soft_complete_board=not board_already_completed,
            details=details,
        )

    # Quiet recognition only — production landed-task guard short-circuits
    # re-implementation from product presence without rewriting durable state.
    # Do not soft-complete here: file presence alone can predate real work.
    return ProgressRecoveryDecision(
        task_id=str(task_id),
        action="recognize_landed_outputs",
        reason="declared_outputs_present_on_merge_target",
        reset_attempt_budget=False,
        clear_diagnostics=False,
        reclaim_dead_lifecycle=False,
        treat_as_landed_outputs=True,
        soft_complete_board=False,
        details=details,
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
    "DEFAULT_LANDED_REVIEW_DEFER_SECONDS",
    "DEFAULT_OPEN_WORK_RESET_COOLDOWN_SECONDS",
    "LANDED_REVIEW_PENDING_MARKERS",
    "LIFECYCLE_RECLAIM_MARKERS",
    "OPEN_WORK_BUDGET_EXHAUSTION_MARKERS",
    "PROGRESS_RECOVERY_FAILURE_MARKERS",
    "DeclaredOutputPresence",
    "ProgressRecoveryDecision",
    "declared_output_presence",
    "operator_landed_binding_payload",
    "should_recover_stalled_task",
]
