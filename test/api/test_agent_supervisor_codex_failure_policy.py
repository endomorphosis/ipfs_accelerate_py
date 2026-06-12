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
    assert validation.needs_rescue
    assert validation.reason == "main_apply_validation_failed_rolled_back"
    assert regression.failed_validation
    assert regression.needs_rescue
    assert regression.reason == "target_metric_regression"


def test_applied_patch_with_failed_validation_is_rescueable_failure():
    outcome = classify_codex_program_outcome(
        codex_exec_status="success",
        patch_status="applied_to_main",
        main_apply_status="applied",
        validation_report={
            "main_apply_validation_status": "failed",
            "status": "failed",
        },
    )

    assert outcome.failed_validation
    assert outcome.needs_rescue
    assert outcome.reason == "main_apply_validation_failed"


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
    assert exhausted.needs_rescue
    assert exhausted.reason == "main_apply_baseline_validation_failed"


def test_target_metric_unavailable_requeues_then_rescues():
    unavailable = classify_codex_program_outcome(
        codex_exec_status="success",
        patch_status="main_apply_target_metric_unavailable_rolled_back",
        validation_report={"target_metric_status": "unavailable"},
        transient_failure_count=1,
        max_transient_failures=3,
    )
    exhausted = classify_codex_program_outcome(
        codex_exec_status="success",
        patch_status="main_apply_target_metric_unavailable_rolled_back",
        validation_report={"target_metric_status": "unavailable"},
        transient_failure_count=3,
        max_transient_failures=3,
    )

    assert unavailable.requeue
    assert unavailable.reason == "target_metric_unavailable"
    assert not unavailable.needs_rescue
    assert exhausted.failed_validation
    assert exhausted.needs_rescue
    assert exhausted.reason == "target_metric_unavailable"
