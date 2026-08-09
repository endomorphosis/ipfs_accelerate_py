"""Tests for deterministic same-attempt implementation auto-rescue."""

from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.todo_daemon.diagnostics import (
    summarize_test_failure,
)
from ipfs_accelerate_py.agent_supervisor.validation.implementation_auto_rescue import (
    AutoRescueAction,
    build_inline_provider_rescue_prompt,
    plan_automatic_implementation_rescue,
)


def test_summarize_test_failure_prefers_assertion_over_banner() -> None:
    output = """
============================= test session starts ==============================
collected 1 item

test/api/test_foo.py F                                                   [100%]

=================================== FAILURES ===================================
_______________________________ test_provider_surfaces _________________________

    def test_provider_surfaces():
>       assert surfaces["count"] >= 1
E       AssertionError: assert 0 >= 1

test/api/test_foo.py:12: AssertionError
=========================== short test summary info ============================
FAILED test/api/test_foo.py::test_provider_surfaces - AssertionError: assert 0 >= 1
============================== 1 failed in 0.12s ===============================
"""
    summary = summarize_test_failure(output)
    assert "test/api/test_foo.py::test_provider_surfaces" in summary["failed_tests"]
    head = summary["failure_head"]
    assert "AssertionError" in head
    assert "short test summary info" not in head
    assert "assert 0 >= 1" in head


def test_summarize_test_failure_quiet_mode_still_extracts_failed_node() -> None:
    output = """
F                                                                        [100%]
=========================== short test summary info ============================
FAILED external/ipfs_accelerate/test/api/test_agent_supervisor_dcr_provider_surface_health.py::test_codec - AssertionError: missing providers
"""
    summary = summarize_test_failure(output)
    assert any("test_codec" in item for item in summary["failed_tests"])
    assert "AssertionError" in summary["failure_head"]
    assert "missing providers" in summary["failure_head"]


def test_plan_stage_and_revalidate_for_empty_patch_with_dirty_outputs() -> None:
    plan = plan_automatic_implementation_rescue(
        validation_result={
            "passed": False,
            "reason": "proposal_gate_failed",
            "error": "proposal_validation_failed",
            "finding_codes": ["empty_patch", "expected_output_ignored_or_unstaged"],
            "failure_review": {
                "decision": "guide_rescue",
                "reason_codes": ["proposal_gate_failed", "empty_or_no_change"],
                "finding_codes": ["empty_patch", "expected_output_ignored_or_unstaged"],
            },
        },
        expected_outputs=(
            "external/ipfs_accelerate/ipfs_accelerate_py/agent_supervisor/analysis/provider_surface_health.py",
            "data/agent_supervisor/deterministic_contract_repair/provider-surfaces.json",
        ),
        expected_outputs_present_on_disk=True,
        dirty_in_scope_paths=(
            "data/agent_supervisor/deterministic_contract_repair/provider-surfaces.json",
        ),
    )
    assert plan.action is AutoRescueAction.STAGE_AND_REVALIDATE
    assert plan.reason == "stage_declared_outputs_and_revalidate"


def test_plan_inline_provider_rescue_for_validation_command_failed() -> None:
    plan = plan_automatic_implementation_rescue(
        validation_result={
            "passed": False,
            "reason": "declared_validation_failed",
            "error": "validation_command_failed",
            "failed_commands": [
                "python3 -m pytest -q external/ipfs_accelerate/test/api/test_foo.py"
            ],
            "failure_head": "E   AssertionError: missing providers",
            "failure_review": {
                "decision": "guide_rescue",
                "reason_codes": ["validation_command_failed"],
                "failed_commands": [
                    "python3 -m pytest -q external/ipfs_accelerate/test/api/test_foo.py"
                ],
                "next_attempt_prompt_addendum": "Re-run and fix the pytest command.",
            },
        },
        expected_outputs=("external/ipfs_accelerate/test/api/test_foo.py",),
        expected_outputs_present_on_disk=True,
        allow_provider_rescue=True,
    )
    assert plan.action is AutoRescueAction.INLINE_PROVIDER_RESCUE
    assert "validation" in plan.reason


def test_plan_refuses_hard_deny_and_exhausted_budget() -> None:
    hard = plan_automatic_implementation_rescue(
        validation_result={
            "passed": False,
            "failure_review": {
                "decision": "reject",
                "reason_codes": ["hard_deny_findings"],
                "finding_codes": ["secret_change_forbidden"],
            },
        },
        expected_outputs_present_on_disk=True,
    )
    assert hard.action is AutoRescueAction.NONE

    exhausted = plan_automatic_implementation_rescue(
        validation_result={
            "passed": False,
            "error": "validation_command_failed",
            "failure_review": {
                "decision": "guide_rescue",
                "reason_codes": ["validation_command_failed"],
            },
        },
        expected_outputs_present_on_disk=True,
        stage_rescue_used=True,
        provider_rescue_passes_used=1,
        already_auto_rescued=True,
    )
    assert exhausted.action is AutoRescueAction.NONE


def test_inline_provider_rescue_prompt_includes_failure_evidence() -> None:
    prompt = build_inline_provider_rescue_prompt(
        base_prompt="Implement DCR-013 outputs.",
        validation_result={
            "next_attempt_prompt_addendum": "Prior attempt failure review (guide_rescue).",
            "failed_commands": ["python3 -m pytest -q test_foo.py"],
            "failed_tests": ["test_foo.py::test_codec"],
            "failure_head": "E   AssertionError: missing providers",
            "failure_review": {
                "decision": "guide_rescue",
                "failed_commands": ["python3 -m pytest -q test_foo.py"],
            },
        },
    )
    assert "Automatic same-attempt validation rescue" in prompt
    assert "Prior attempt failure review" in prompt
    assert "test_foo.py::test_codec" in prompt
    assert "AssertionError: missing providers" in prompt
    assert prompt.startswith("Implement DCR-013 outputs.")
