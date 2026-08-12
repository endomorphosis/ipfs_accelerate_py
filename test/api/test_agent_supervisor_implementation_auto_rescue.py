"""Tests for deterministic same-attempt implementation auto-rescue."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon
from ipfs_accelerate_py.agent_supervisor.todo_daemon.diagnostics import (
    summarize_test_failure,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon,
    PortalTask,
)
from ipfs_accelerate_py.agent_supervisor.validation.implementation_auto_rescue import (
    AutoRescueAction,
    build_inline_provider_rescue_prompt,
    derive_materialize_commands,
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


def test_plan_stage_after_proposal_accept_when_outputs_incomplete() -> None:
    """PTR-style: proposal accepted, residual review says outputs incomplete."""

    plan = plan_automatic_implementation_rescue(
        validation_result={
            "passed": False,
            "reason": "declared_validation_failed",
            "error": "validation_command_failed",
            "failed_commands": ["cargo test --locked --manifest-path ..."],
            "proposal_gate": {"accepted": True},
            "failure_review": {
                "decision": "guide_rescue",
                "reason_codes": [
                    "incomplete_expected_outputs",
                    "large_or_undeclared_refactor",
                ],
                "missing_expected_outputs": [
                    "external/ipfs_datasets/ipfs_datasets_py/processors/groth16_backend/RUST_SETUP.md",
                    "external/ipfs_datasets/ipfs_datasets_py/processors/groth16_backend/WIRE_FORMAT.md",
                ],
                "failed_commands": ["cargo test --locked --manifest-path ..."],
            },
        },
        expected_outputs=(
            "external/ipfs_datasets/ipfs_datasets_py/processors/groth16_backend/RUST_SETUP.md",
            "external/ipfs_datasets/ipfs_datasets_py/processors/groth16_backend/WIRE_FORMAT.md",
        ),
        expected_outputs_present_on_disk=True,
        dirty_in_scope_paths=(
            "external/ipfs_datasets/ipfs_datasets_py/processors/groth16_backend/RUST_SETUP.md",
        ),
        allow_provider_rescue=True,
    )
    assert plan.action is AutoRescueAction.STAGE_AND_REVALIDATE
    assert plan.reason == "stage_declared_outputs_and_revalidate"

    after_stage = plan_automatic_implementation_rescue(
        validation_result={
            "passed": False,
            "reason": "declared_validation_failed",
            "error": "validation_command_failed",
            "failed_commands": ["cargo test --locked --manifest-path ..."],
            "proposal_gate": {"accepted": True},
            "failure_review": {
                "decision": "guide_rescue",
                "reason_codes": ["incomplete_expected_outputs"],
                "failed_commands": ["cargo test --locked --manifest-path ..."],
            },
        },
        expected_outputs=(
            "external/ipfs_datasets/ipfs_datasets_py/processors/groth16_backend/RUST_SETUP.md",
        ),
        expected_outputs_present_on_disk=True,
        stage_rescue_used=True,
        allow_provider_rescue=True,
    )
    assert after_stage.action is AutoRescueAction.INLINE_PROVIDER_RESCUE


def test_derive_materialize_commands_from_validate_cli() -> None:
    commands = derive_materialize_commands(
        (
            "PYTHONPATH=external/ipfs_accelerate python3 -m "
            "external.ipfs_accelerate.ipfs_accelerate_py.agent_supervisor.analysis."
            "deterministic_desktop_expectations validate --workspace . "
            "--artifact data/agent_supervisor/deterministic_contract_repair/"
            "desktop-expectations.json",
        )
    )
    assert commands
    assert any(" materialize " in command for command in commands)
    assert all(" validate " not in command for command in commands)


def test_plan_materialize_when_expected_artifact_missing() -> None:
    plan = plan_automatic_implementation_rescue(
        validation_result={
            "passed": False,
            "reason": "proposal_gate_failed",
            "failure_review": {
                "decision": "guide_rescue",
                "reason_codes": [
                    "incomplete_expected_outputs",
                    "proposal_gate_failed",
                ],
                "finding_codes": ["expected_output_ignored_or_unstaged"],
                "missing_expected_outputs": [
                    "data/agent_supervisor/deterministic_contract_repair/"
                    "desktop-expectations.json"
                ],
            },
        },
        expected_outputs=(
            "data/agent_supervisor/deterministic_contract_repair/"
            "desktop-expectations.json",
            "external/ipfs_accelerate/ipfs_accelerate_py/agent_supervisor/"
            "analysis/deterministic_desktop_expectations.py",
        ),
        validation_commands=(
            "python3 -m pkg.mod validate --workspace . --artifact "
            "data/agent_supervisor/deterministic_contract_repair/"
            "desktop-expectations.json",
        ),
        missing_expected_outputs=(
            "data/agent_supervisor/deterministic_contract_repair/"
            "desktop-expectations.json",
        ),
        expected_outputs_present_on_disk=False,
    )
    assert plan.action is AutoRescueAction.MATERIALIZE_AND_STAGE
    assert plan.materialize_commands
    assert "desktop-expectations.json" in " ".join(plan.missing_expected_outputs)


def test_plan_provider_rescue_after_stage_for_residual_incomplete() -> None:
    plan = plan_automatic_implementation_rescue(
        validation_result={
            "passed": False,
            "reason": "proposal_gate_failed",
            "failure_review": {
                "decision": "guide_rescue",
                "reason_codes": [
                    "incomplete_expected_outputs",
                    "proposal_gate_failed",
                ],
                "finding_codes": ["expected_output_ignored_or_unstaged"],
                "missing_expected_outputs": [
                    "data/agent_supervisor/deterministic_contract_repair/"
                    "desktop-expectations.json"
                ],
            },
        },
        expected_outputs=(
            "data/agent_supervisor/deterministic_contract_repair/"
            "desktop-expectations.json",
        ),
        stage_rescue_used=True,
        materialize_rescue_used=True,
        allow_provider_rescue=True,
        expected_outputs_present_on_disk=False,
        missing_expected_outputs=(
            "data/agent_supervisor/deterministic_contract_repair/"
            "desktop-expectations.json",
        ),
    )
    assert plan.action is AutoRescueAction.INLINE_PROVIDER_RESCUE
    assert "residual" in plan.reason or "incomplete" in plan.reason


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
        materialize_rescue_used=True,
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


def _inline_rescue_test_daemon(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[PortalImplementationDaemon, list[tuple[str, dict[str, object]]]]:
    daemon = object.__new__(PortalImplementationDaemon)
    daemon.implementation_timeout = 60
    daemon.implementation_max_timeout = 60
    events: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        daemon,
        "_expected_outputs_present_on_disk",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        daemon,
        "_dirty_in_scope_declared_output_paths",
        lambda *_args, **_kwargs: (),
    )
    monkeypatch.setattr(
        daemon,
        "_record_event",
        lambda name, payload: events.append((name, dict(payload))),
    )
    monkeypatch.setattr(
        daemon,
        "_ensure_implementation_checkpoint_dir",
        lambda _task: tmp_path / "checkpoint",
    )
    monkeypatch.setattr(
        daemon,
        "_implementation_process_environment",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        daemon,
        "_stage_declared_candidate_outputs",
        lambda *_args, **_kwargs: (),
    )
    monkeypatch.setattr(
        daemon,
        "_run_validation_with_candidate_binding",
        lambda *_args, **_kwargs: {"passed": True},
    )
    monkeypatch.setattr(
        daemon,
        "_apply_implementation_failure_review",
        lambda **kwargs: dict(kwargs["validation_result"]),
    )
    return daemon, events


def _run_inline_rescue(
    daemon: PortalImplementationDaemon,
    tmp_path: Path,
    command: list[str],
) -> dict[str, object]:
    task = PortalTask(
        task_id="RESCUE-001",
        title="repair validation",
        status="in_progress",
        completion="validation passes",
        priority="high",
        track="test",
        outputs=["result.txt"],
    )
    return daemon._automatic_implementation_rescue(
        task=task,
        attempt=1,
        workspace_path=tmp_path,
        branch_name="agent/rescue-001",
        baseline_ref="a" * 40,
        validation_result={
            "passed": False,
            "reason": "declared_validation_failed",
            "error": "validation_command_failed",
            "failed_commands": ["python3 -m pytest -q test_result.py"],
            "failure_review": {
                "decision": "guide_rescue",
                "reason_codes": ["validation_command_failed"],
            },
        },
        log_path=tmp_path / "implementation.log",
        state=None,
        command=command,
        base_prompt="repair the implementation",
    )


def test_inline_provider_rescue_refuses_prompt_bound_control_plane_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, _events = _inline_rescue_test_daemon(tmp_path, monkeypatch)
    accepted_path = "/proc/self/fd/71"
    daemon._scoped_control_plane_launch = SimpleNamespace(
        descriptor=71,
        executable_path=accepted_path,
    )
    daemon._scoped_recovery_control_plane_launches = {
        "unrelated": SimpleNamespace(
            descriptor=72,
            executable_path="/proc/self/fd/72",
        )
    }
    command = [sys.executable, "-I", accepted_path, "--workspace", str(tmp_path)]
    monkeypatch.setattr(
        implementation_daemon,
        "run_process_group_stream",
        lambda *_args, **_kwargs: pytest.fail(
            "prompt-bound control-plane command must not run inline rescue"
        ),
    )

    result = _run_inline_rescue(daemon, tmp_path, command)

    assert result["passed"] is False
    assert result["auto_rescue_terminal"] is True
    assert result["auto_rescue"]["provider_passes"] == 0


def test_inline_provider_rescue_fails_closed_on_ambiguous_sealed_fd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, events = _inline_rescue_test_daemon(tmp_path, monkeypatch)
    accepted_path = "/proc/self/fd/71"
    daemon._scoped_control_plane_launch = SimpleNamespace(
        descriptor=71,
        executable_path=accepted_path,
    )
    daemon._scoped_recovery_control_plane_launches = {
        "ambiguous": SimpleNamespace(
            descriptor=72,
            executable_path=accepted_path,
        )
    }
    command = [sys.executable, "-I", accepted_path, "--workspace", str(tmp_path)]
    monkeypatch.setattr(
        implementation_daemon,
        "run_process_group_stream",
        lambda *_args, **_kwargs: pytest.fail(
            "ambiguous control-plane authority must not launch a provider"
        ),
    )

    result = _run_inline_rescue(daemon, tmp_path, command)

    assert result["passed"] is False
    assert result["auto_rescue_terminal"] is True
    assert not any(name.endswith("provider_started") for name, _payload in events)


def test_inline_provider_rescue_keeps_unsealed_command_without_pass_fds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, _events = _inline_rescue_test_daemon(tmp_path, monkeypatch)
    daemon._scoped_control_plane_launch = None
    daemon._scoped_recovery_control_plane_launches = {}
    command = ["/opt/providers/grok", "--model", "grok-4.5"]
    calls: list[dict[str, object]] = []

    def fake_stream(run_command, **kwargs):
        calls.append(dict(kwargs))
        return subprocess.CompletedProcess(run_command, 0)

    monkeypatch.setattr(
        implementation_daemon,
        "run_process_group_stream",
        fake_stream,
    )

    result = _run_inline_rescue(daemon, tmp_path, command)

    assert result["passed"] is True
    assert len(calls) == 1
    assert "pass_fds" not in calls[0]
