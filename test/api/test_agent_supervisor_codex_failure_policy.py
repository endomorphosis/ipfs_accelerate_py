from ipfs_accelerate_py.agent_supervisor.codex_failure_policy import (
    classify_codex_program_outcome,
)


def test_completed_patch_status_completes_queue_item():
    outcome = classify_codex_program_outcome(
        codex_exec_status="success",
        patch_status="applied_to_main",
        main_apply_status="applied",
    )

    assert outcome.completed
    assert outcome.reason is None


def test_lock_timeout_requeues_until_transient_budget_is_exhausted():
    outcome = classify_codex_program_outcome(
        codex_exec_status="success",
        patch_status="main_apply_lock_timeout",
        main_apply_status="lock_timeout",
        transient_failure_count=2,
        max_transient_failures=3,
    )

    assert outcome.requeue
    assert outcome.reason == "main_apply_lock_timeout"

    exhausted = classify_codex_program_outcome(
        codex_exec_status="success",
        patch_status="main_apply_lock_timeout",
        main_apply_status="lock_timeout",
        transient_failure_count=3,
        max_transient_failures=3,
    )

    assert exhausted.failed_validation
    assert exhausted.reason == "main_apply_lock_timeout"


def test_stale_or_dirty_apply_failures_are_transient():
    for patch_status in (
        "main_apply_dirty_target",
        "main_apply_dirty_target_repair_failed",
        "main_apply_check_failed_repair_failed",
        "main_apply_failed",
        "awaiting_codex_changes",
    ):
        outcome = classify_codex_program_outcome(
            codex_exec_status="success",
            patch_status=patch_status,
            transient_failure_count=0,
            max_transient_failures=3,
        )

        assert outcome.requeue
        assert outcome.reason == patch_status


def test_validation_failures_and_metric_regressions_remain_terminal():
    validation = classify_codex_program_outcome(
        codex_exec_status="success",
        patch_status="main_apply_validation_failed_rolled_back",
        validation_report={"status": "failed"},
    )
    regression = classify_codex_program_outcome(
        codex_exec_status="success",
        patch_status="main_apply_target_metric_regression_rolled_back",
        validation_report={"target_metric_status": "regressed"},
    )

    assert validation.failed_validation
    assert validation.reason == "main_apply_validation_failed_rolled_back"
    assert regression.failed_validation
    assert regression.reason == "target_metric_regression"


def test_baseline_validation_failures_are_transient_until_budget_exhausted():
    outcome = classify_codex_program_outcome(
        codex_exec_status="success",
        patch_status="main_apply_baseline_validation_failed_rolled_back",
        transient_failure_count=0,
        max_transient_failures=3,
    )
    exhausted = classify_codex_program_outcome(
        codex_exec_status="success",
        patch_status="main_apply_baseline_validation_failed_rolled_back",
        transient_failure_count=3,
        max_transient_failures=3,
    )

    assert outcome.requeue
    assert outcome.reason == "main_apply_baseline_validation_failed"
    assert exhausted.failed_validation
    assert exhausted.reason == "main_apply_baseline_validation_failed"
