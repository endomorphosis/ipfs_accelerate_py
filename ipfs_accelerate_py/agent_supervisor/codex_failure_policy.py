"""Reusable Codex apply/validation outcome policy for agent supervisors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


COMPLETED_PATCH_STATUSES = frozenset(
    {
        "created",
        "applied_to_main",
        "main_apply_no_merged_delta",
    }
)

TRANSIENT_PATCH_STATUSES = frozenset(
    {
        "awaiting_codex_changes",
        "main_apply_check_failed",
        "main_apply_check_failed_repair_failed",
        "main_apply_dirty_target",
        "main_apply_dirty_target_repair_failed",
        "main_apply_failed",
        "main_apply_lock_timeout",
        "patch_generation_failed",
        "worktree_unavailable",
        "submodule_update_failed",
        "submodule_conflict",
        "submodule_checkout_failed",
        "merge_conflict_submodule",
    }
)

TRANSIENT_MAIN_APPLY_STATUSES = frozenset({"lock_timeout", "no_changes"})

TERMINAL_VALIDATION_PATCH_PREFIXES = ("main_apply_validation_failed",)

TERMINAL_TARGET_METRIC_REGRESSION_PREFIX = "main_apply_target_metric_regression"

TARGET_METRIC_UNAVAILABLE_REASONS = frozenset(
    {
        "target_metric_unavailable",
        "main_apply_target_metric_unavailable",
        "main_apply_target_metric_unavailable_rolled_back",
    }
)


@dataclass(frozen=True)
class CodexProgramOutcome:
    """Decision for finalizing claimed program-synthesis TODOs."""

    action: str
    reason: Optional[str] = None
    rescue: bool = False

    @property
    def completed(self) -> bool:
        return self.action == "completed"

    @property
    def requeue(self) -> bool:
        return self.action == "requeue"

    @property
    def failed_validation(self) -> bool:
        return self.action == "failed_validation"

    @property
    def needs_rescue(self) -> bool:
        return self.failed_validation and self.rescue


def _normalized(value: Any) -> str:
    return str(value or "").strip().lower()


def _transient_budget_exhausted(
    *,
    transient_failure_count: int,
    max_transient_failures: int,
) -> bool:
    return int(transient_failure_count) >= max(0, int(max_transient_failures))


def _validation_failed(report: Mapping[str, Any]) -> bool:
    return any(
        _normalized(report.get(key)) == "failed"
        for key in (
            "status",
            "main_apply_validation_status",
            "validation_status",
        )
    )


def classify_codex_program_outcome(
    *,
    codex_exec_status: str,
    patch_status: str | None,
    main_apply_status: str | None = None,
    validation_report: Mapping[str, Any] | None = None,
    transient_failure_count: int = 0,
    max_transient_failures: int = 3,
) -> CodexProgramOutcome:
    """Classify a Codex packet outcome for queue finalization.

    The policy separates real candidate-quality failures from infrastructure and
    concurrency noise. Stale diffs, dirty target files, apply-lock timeouts, and
    empty/no-change packets are retried up to a bounded transient budget. Actual
    validation failures and target-metric regressions remain terminal so rescue
    TODOs can be generated from their evidence.
    """

    exec_status = _normalized(codex_exec_status)
    patch = _normalized(patch_status)
    main_apply = _normalized(main_apply_status)
    report = dict(validation_report or {})

    target_metric_status = _normalized(report.get("target_metric_status"))
    report_reason = _normalized(
        report.get("failure_reason")
        or report.get("reason")
        or report.get("status_reason")
        or report.get("error")
    )
    if (
        patch in TARGET_METRIC_UNAVAILABLE_REASONS
        or report_reason in TARGET_METRIC_UNAVAILABLE_REASONS
        or target_metric_status == "unavailable"
    ):
        reason = "target_metric_unavailable"
        if _transient_budget_exhausted(
            transient_failure_count=transient_failure_count,
            max_transient_failures=max_transient_failures,
        ):
            return CodexProgramOutcome("failed_validation", reason, rescue=True)
        return CodexProgramOutcome("requeue", reason)

    if patch.startswith(TERMINAL_TARGET_METRIC_REGRESSION_PREFIX) or target_metric_status == "regressed":
        return CodexProgramOutcome("failed_validation", "target_metric_regression", rescue=True)

    if patch.startswith("main_apply_baseline_validation_failed"):
        reason = "main_apply_baseline_validation_failed"
        if _transient_budget_exhausted(
            transient_failure_count=transient_failure_count,
            max_transient_failures=max_transient_failures,
        ):
            return CodexProgramOutcome("failed_validation", reason, rescue=True)
        return CodexProgramOutcome("requeue", reason)

    if any(patch.startswith(prefix) for prefix in TERMINAL_VALIDATION_PATCH_PREFIXES):
        return CodexProgramOutcome(
            "failed_validation",
            patch or "main_apply_validation_failed",
            rescue=True,
        )

    if _validation_failed(report):
        return CodexProgramOutcome(
            "failed_validation",
            "main_apply_validation_failed",
            rescue=True,
        )

    if patch in COMPLETED_PATCH_STATUSES or main_apply == "applied":
        return CodexProgramOutcome("completed")

    if exec_status == "transient_failure":
        reason = "codex_exec_transient_failure"
        if _transient_budget_exhausted(
            transient_failure_count=transient_failure_count,
            max_transient_failures=max_transient_failures,
        ):
            return CodexProgramOutcome("failed_validation", reason, rescue=True)
        return CodexProgramOutcome("requeue", reason)

    if patch in TRANSIENT_PATCH_STATUSES or main_apply in TRANSIENT_MAIN_APPLY_STATUSES:
        reason = patch or main_apply or "transient_apply_failure"
        if _transient_budget_exhausted(
            transient_failure_count=transient_failure_count,
            max_transient_failures=max_transient_failures,
        ):
            return CodexProgramOutcome("failed_validation", reason, rescue=True)
        return CodexProgramOutcome("requeue", reason)

    if (
        patch.startswith("main_apply_target_metric_")
        and patch.endswith("_rolled_back")
        and "regression" not in patch
    ):
        reason = patch.removeprefix("main_apply_").removesuffix("_rolled_back")
        if _transient_budget_exhausted(
            transient_failure_count=transient_failure_count,
            max_transient_failures=max_transient_failures,
        ):
            return CodexProgramOutcome("failed_validation", reason, rescue=True)
        return CodexProgramOutcome("requeue", reason)

    if exec_status in {"failed", "timeout"}:
        return CodexProgramOutcome("failed_validation", f"codex_exec_{exec_status}", rescue=True)

    if patch:
        return CodexProgramOutcome("failed_validation", patch, rescue=True)
    return CodexProgramOutcome("failed_validation", "patch_not_created", rescue=True)
