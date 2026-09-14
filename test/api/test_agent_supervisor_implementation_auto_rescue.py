"""Tests for deterministic same-attempt implementation auto-rescue."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from ipfs_accelerate_py.agent_supervisor.proof.code_proof_obligations import (
    DiffChangeKind,
)
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
    build_scoped_test_semantic_inventory,
    derive_materialize_commands,
    is_undeclared_helper_path,
    plan_automatic_implementation_rescue,
)


def _secret_test_source(value: str, *, include_assertion: bool = True) -> str:
    source = (
        "def test_secret_redaction():\n"
        f"    password = {value!r}\n"
    )
    if include_assertion:
        source += "    assert redact(password) == '[redacted]'\n"
    return source


def _test_proposal(
    proposal_id: str,
    source: str,
    *,
    path: str = "tests/unit/test_redaction.py",
) -> SimpleNamespace:
    return SimpleNamespace(
        proposal_id=proposal_id,
        changed_paths=(path,),
        candidate_diff=(
            SimpleNamespace(
                before_source="",
                after_source=source,
                new_path=path,
                old_path=path,
            ),
        ),
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


def test_is_undeclared_helper_path_recognizes_scratch_files() -> None:
    expected = ("swissknife/src/services/gui-optimizer/cli.ts",)
    assert is_undeclared_helper_path("tmp-vgo-062-write-evidence.mts", expected)
    assert is_undeclared_helper_path("swissknife/_run_registry.py", expected)
    assert is_undeclared_helper_path("DELETE_ME_helper.py", expected)
    assert is_undeclared_helper_path("scripts/vgo060-selfcheck.py", expected)
    assert not is_undeclared_helper_path(
        "swissknife/src/services/gui-optimizer/cli.ts",
        expected,
    )
    assert not is_undeclared_helper_path(
        "swissknife/src/services/gui-optimizer/targets/agent-supervisor.ts",
        expected,
    )


def test_plan_strips_helper_only_scope_denials_when_outputs_exist() -> None:
    plan = plan_automatic_implementation_rescue(
        validation_result={
            "passed": False,
            "reason": "scope_adjudication_failed",
            "failure_review": {
                "decision": "reject",
                "reason_codes": ["scope_expansion_denied"],
                "finding_codes": ["path_outside_scope"],
                "denied_paths": [
                    "tmp-vgo-062-write-evidence.mts",
                    "swissknife/_run_selfcheck.py",
                ],
            },
        },
        expected_outputs=(
            "swissknife/src/services/gui-optimizer/targets/agent-supervisor.ts",
            "swissknife/test/unit/services/gui-optimizer/agent-supervisor-baseline.test.ts",
        ),
        expected_outputs_present_on_disk=True,
    )
    assert plan.action is AutoRescueAction.STRIP_DENIED_HELPERS
    assert plan.reason == "strip_undeclared_helper_paths"
    assert plan.denied_helper_paths == (
        "swissknife/_run_selfcheck.py",
        "tmp-vgo-062-write-evidence.mts",
    )

    mixed = plan_automatic_implementation_rescue(
        validation_result={
            "passed": False,
            "failure_review": {
                "decision": "reject",
                "reason_codes": ["scope_expansion_denied"],
                "finding_codes": ["path_outside_scope"],
                "denied_paths": [
                    "tmp-vgo-060-helper.py",
                    "swissknife/src/unrelated/secret.ts",
                ],
            },
        },
        expected_outputs=("scripts/gui-opt",),
        expected_outputs_present_on_disk=True,
    )
    assert mixed.action is AutoRescueAction.NONE
    assert mixed.reason in {"hard_deny_or_reject", "hard_deny_reason_codes"}

    after_strip = plan_automatic_implementation_rescue(
        validation_result={
            "passed": False,
            "failure_review": {
                "decision": "reject",
                "reason_codes": ["scope_expansion_denied"],
                "denied_paths": ["tmp-vgo-062-write-evidence.mts"],
            },
        },
        expected_outputs=("scripts/gui-opt",),
        expected_outputs_present_on_disk=True,
        strip_helpers_used=True,
    )
    assert after_strip.action is AutoRescueAction.NONE


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


def _scoped_test_secret_failure_result(
    *,
    path: str = "tests/unit/test_redaction.py",
    private_key_absence_verified: bool = True,
    credential_assignment_only: bool = True,
    out_of_scope_paths: tuple[str, ...] = (),
    finding_codes: tuple[str, ...] = ("secret_change_forbidden",),
) -> dict[str, object]:
    in_scope_paths = () if out_of_scope_paths else (path,)
    rejected_proposal = _test_proposal(
        "proposal:rejected-secret",
        _secret_test_source("s3cret-value"),
        path=path,
    )
    semantic_inventory = build_scoped_test_semantic_inventory(
        rejected_proposal,
        (path,),
        concrete_secret_value=lambda _value: True,
    )
    assert semantic_inventory is not None
    return {
        "passed": False,
        "reason": "proposal_gate_failed",
        "error": "proposal_validation_failed",
        "proposal_gate": {
            "accepted": False,
            "proposal_id": "proposal:rejected-secret",
            "receipt_id": "receipt:rejected-secret",
            "reason_codes": list(finding_codes),
        },
        "failure_review": {
            "decision": "reject",
            "reason_codes": ["hard_deny_findings", "proposal_gate_failed"],
            "finding_codes": list(finding_codes),
        },
        "secret_change_scope_examination": {
            "proposal_id": "proposal:rejected-secret",
            "finding_code": "secret_change_forbidden",
            "examined_paths": [path],
            "in_scope_paths": list(in_scope_paths),
            "out_of_scope_paths": list(out_of_scope_paths),
            "scoped_python_test_source_paths": [path],
            "scope_classification": (
                "out_of_scope" if out_of_scope_paths else "in_scope"
            ),
            "candidate_diff_path_coverage_complete": True,
            "private_key_material_detected": (
                not private_key_absence_verified
            ),
            "private_key_material_absence_verified": (
                private_key_absence_verified
            ),
            "credential_assignment_only": credential_assignment_only,
            "test_semantic_inventory": semantic_inventory,
            "secret_policy_overridden": False,
        },
    }


def test_plan_allows_one_repair_pass_for_rejected_scoped_test_secret() -> None:
    unbound = plan_automatic_implementation_rescue(
        validation_result=_scoped_test_secret_failure_result(),
        expected_outputs=("tests/unit/test_redaction.py",),
        allow_provider_rescue=True,
    )
    assert unbound.action is AutoRescueAction.NONE

    plan = plan_automatic_implementation_rescue(
        validation_result=_scoped_test_secret_failure_result(),
        expected_outputs=("tests/unit/test_redaction.py",),
        allow_provider_rescue=True,
        provider_rescue_passes_used=0,
        accepted_effect_count=0,
        merge_effect_count=0,
    )

    assert plan.action is AutoRescueAction.REMEDIATE_SCOPED_TEST_SECRET
    assert plan.remediation_paths == ("tests/unit/test_redaction.py",)
    assert plan.prior_proposal_id == "proposal:rejected-secret"
    assert plan.prior_receipt_id == "receipt:rejected-secret"
    assert plan.accepted_effect_count == 0
    assert plan.merge_effect_count == 0
    assert plan.max_provider_rescue_passes == 1
    assert plan.prior_test_semantic_inventory is not None
    assert "s3cret-value" not in str(plan.to_record())


def test_plan_refuses_scoped_secret_rescue_without_bound_test_inventory() -> None:
    result = _scoped_test_secret_failure_result()
    examination = result["secret_change_scope_examination"]
    assert isinstance(examination, dict)
    examination.pop("test_semantic_inventory")

    plan = plan_automatic_implementation_rescue(
        validation_result=result,
        expected_outputs=("tests/unit/test_redaction.py",),
        allow_provider_rescue=True,
        provider_rescue_passes_used=0,
        accepted_effect_count=0,
        merge_effect_count=0,
    )

    assert plan.action is AutoRescueAction.NONE


@pytest.mark.parametrize(
    "result_kwargs,planner_kwargs",
    [
        ({"path": "src/redaction.py"}, {}),
        ({"out_of_scope_paths": ("tests/unit/test_redaction.py",)}, {}),
        ({"private_key_absence_verified": False}, {}),
        ({"credential_assignment_only": False}, {}),
        (
            {
                "finding_codes": (
                    "secret_change_forbidden",
                    "test_weakening_forbidden",
                )
            },
            {},
        ),
        ({}, {"accepted_effect_count": 1}),
        ({}, {"merge_effect_count": 1}),
        ({}, {"provider_rescue_passes_used": 1}),
    ],
)
def test_plan_refuses_unsafe_or_repeated_scoped_test_secret_remediation(
    result_kwargs: dict[str, object],
    planner_kwargs: dict[str, int],
) -> None:
    result = _scoped_test_secret_failure_result(**result_kwargs)
    # Production paths are deliberately not classified as scoped test source.
    if result_kwargs.get("path") == "src/redaction.py":
        examination = result["secret_change_scope_examination"]
        assert isinstance(examination, dict)
        examination["scoped_python_test_source_paths"] = []

    kwargs = {
        "provider_rescue_passes_used": 0,
        "accepted_effect_count": 0,
        "merge_effect_count": 0,
        **planner_kwargs,
    }
    plan = plan_automatic_implementation_rescue(
        validation_result=result,
        expected_outputs=("tests/unit/test_redaction.py",),
        allow_provider_rescue=True,
        **kwargs,
    )

    assert plan.action is AutoRescueAction.NONE


def test_scoped_test_secret_rescue_prompt_changes_strategy_without_value() -> None:
    failure = _scoped_test_secret_failure_result()
    plan = plan_automatic_implementation_rescue(
        validation_result=failure,
        expected_outputs=("tests/unit/test_redaction.py",),
        allow_provider_rescue=True,
        accepted_effect_count=0,
        merge_effect_count=0,
    )

    prompt = build_inline_provider_rescue_prompt(
        base_prompt="Implement the redaction trace test.",
        validation_result=failure,
        auto_rescue_plan=plan,
    )

    assert "Scoped test-secret remediation" in prompt
    assert "prior proposal remains rejected" in prompt
    assert "different proposal" in prompt
    assert "do not weaken" in prompt
    assert "tests/unit/test_redaction.py" in prompt
    assert "actual-secret-from-candidate" not in prompt


def test_secret_scope_examination_proves_assignment_only_not_private_key() -> None:
    path = "tests/unit/test_redaction.py"
    policy = SimpleNamespace(
        policy_id="policy:test-scope",
        path_is_in_scope=lambda candidate: candidate == path,
    )

    def examination_for(source: str) -> dict[str, object]:
        entry = SimpleNamespace(
            before_source="",
            after_source=source,
            change_kind=DiffChangeKind.MODIFY,
            new_path=path,
            old_path=path,
        )
        proposal = SimpleNamespace(
            proposal_id="proposal:scope-examination",
            changed_paths=(path,),
            candidate_diff=(entry,),
        )
        result = SimpleNamespace(
            findings=(
                SimpleNamespace(
                    code=SimpleNamespace(value="secret_change_forbidden"),
                    path=path,
                ),
            ),
            proposal=proposal,
            policy=policy,
        )
        examination = PortalImplementationDaemon._secret_change_scope_examination(
            result
        )
        assert examination is not None
        return examination

    assignment_source = (
        "pass" + "word = \"" + "s3cret" + "-value\"\n"
    )
    assignment = examination_for(assignment_source)
    assert assignment["scoped_python_test_source_paths"] == [path]
    assert assignment["candidate_diff_path_coverage_complete"] is True
    assert assignment["private_key_material_detected"] is False
    assert assignment["private_key_material_absence_verified"] is True
    assert assignment["credential_assignment_only"] is True

    private_key_source = (
        "-----BEGIN " + "PRIVATE KEY-----\nnot-a-key\n-----END PRIVATE KEY-----\n"
    )
    private_key = examination_for(private_key_source)
    assert private_key["private_key_material_detected"] is True
    assert private_key["private_key_material_absence_verified"] is False
    assert private_key["credential_assignment_only"] is False


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
    command = ["/opt/providers/grok", "--model", "grok-4.6"]
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


def test_scoped_test_secret_remediation_runs_once_then_fully_revalidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, events = _inline_rescue_test_daemon(tmp_path, monkeypatch)
    daemon._scoped_control_plane_launch = None
    daemon._scoped_recovery_control_plane_launches = {}
    provider_calls: list[dict[str, object]] = []
    validation_calls: list[dict[str, object]] = []

    def fake_stream(run_command, **kwargs):
        provider_calls.append({"command": list(run_command), **dict(kwargs)})
        return subprocess.CompletedProcess(run_command, 0)

    replacement = SimpleNamespace(
        accepted=True,
        proposal=_test_proposal(
            "proposal:replacement-secret",
            _secret_test_source("literal-secret-canary"),
        ),
    )

    def fake_revalidate(*_args, **kwargs):
        validation_calls.append(dict(kwargs))
        return {
            "attempted": True,
            "passed": True,
            "returncode": 0,
            "proposal_validation": replacement,
            "proposal_gate": {
                "accepted": True,
                "proposal_id": "proposal:replacement-secret",
                "receipt_id": "receipt:replacement-secret",
            },
            "selection": {"scope": "pre_merge"},
        }

    monkeypatch.setattr(
        implementation_daemon,
        "run_process_group_stream",
        fake_stream,
    )
    monkeypatch.setattr(
        daemon,
        "_run_validation_with_candidate_binding",
        fake_revalidate,
    )
    task = PortalTask(
        task_id="RESCUE-SECRET-001",
        title="repair a redaction test canary",
        status="in_progress",
        completion="proposal and tests pass",
        priority="high",
        track="test",
        outputs=["tests/unit/test_redaction.py"],
        validation=["python -m pytest -q tests/unit/test_redaction.py"],
    )

    result = daemon._automatic_implementation_rescue(
        task=task,
        attempt=1,
        workspace_path=tmp_path,
        branch_name="agent/rescue-secret-001",
        baseline_ref="a" * 40,
        validation_result=_scoped_test_secret_failure_result(),
        log_path=tmp_path / "implementation.log",
        state=None,
        command=["/opt/providers/grok", "--model", "grok-4.6"],
        base_prompt="Implement the redaction trace test.",
    )

    assert result["passed"] is True
    assert result["auto_rescue_terminal"] is True
    assert result["auto_rescue"]["provider_passes"] == 1
    assert len(provider_calls) == 1
    assert len(validation_calls) == 1
    # Omitting a prior live proposal forces a fresh proposal-gate run followed
    # by the ordinary full pre-merge validation plan.
    assert validation_calls[0]["proposal_validation"] is None
    binding = result["scoped_test_secret_remediation_binding"]
    assert binding["prior_proposal_id"] == "proposal:rejected-secret"
    assert binding["prior_receipt_id"] == "receipt:rejected-secret"
    assert binding["replacement_proposal_id"] == "proposal:replacement-secret"
    assert binding["replacement_is_fresh"] is True
    assert binding["replacement_gate_accepted"] is True
    assert binding["full_validation_passed"] is True
    started = next(
        payload
        for name, payload in events
        if name == "implementation_auto_rescue_provider_started"
    )
    assert started["strategy_changed"] is True
    assert started["base_prompt_id"] != started["rescue_prompt_id"]
    assert started["plan"]["prior_receipt_id"] == "receipt:rejected-secret"
    assert "actual-secret-from-candidate" not in str(started)


def test_scoped_test_secret_remediation_rejects_unchanged_proposal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, _events = _inline_rescue_test_daemon(tmp_path, monkeypatch)
    daemon._scoped_control_plane_launch = None
    daemon._scoped_recovery_control_plane_launches = {}
    monkeypatch.setattr(
        implementation_daemon,
        "run_process_group_stream",
        lambda command, **_kwargs: subprocess.CompletedProcess(command, 0),
    )
    unchanged = SimpleNamespace(
        accepted=True,
        proposal=_test_proposal(
            "proposal:rejected-secret",
            _secret_test_source("literal-secret-canary"),
        ),
    )
    monkeypatch.setattr(
        daemon,
        "_run_validation_with_candidate_binding",
        lambda *_args, **_kwargs: {
            "attempted": True,
            "passed": True,
            "returncode": 0,
            "proposal_validation": unchanged,
            "proposal_gate": {
                "accepted": True,
                "proposal_id": "proposal:rejected-secret",
                "receipt_id": "receipt:unchanged",
            },
        },
    )
    task = PortalTask(
        task_id="RESCUE-SECRET-002",
        title="repair a redaction test canary",
        status="in_progress",
        completion="proposal and tests pass",
        priority="high",
        track="test",
        outputs=["tests/unit/test_redaction.py"],
        validation=["python -m pytest -q tests/unit/test_redaction.py"],
    )

    result = daemon._automatic_implementation_rescue(
        task=task,
        attempt=1,
        workspace_path=tmp_path,
        branch_name="agent/rescue-secret-002",
        baseline_ref="a" * 40,
        validation_result=_scoped_test_secret_failure_result(),
        log_path=tmp_path / "implementation.log",
        state=None,
        command=["/opt/providers/grok"],
        base_prompt="Implement the redaction trace test.",
    )

    assert result["passed"] is False
    assert result["auto_rescue_terminal"] is True
    assert result["auto_rescue"]["provider_passes"] == 1
    assert result["reason"] == "scoped_test_secret_remediation_proposal_unchanged"
    assert result["scoped_test_secret_remediation_binding"][
        "replacement_is_fresh"
    ] is False


def test_scoped_test_secret_remediation_rejects_deleted_security_assertion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ordinary green gate cannot weaken a test added by the rejection."""

    daemon, _events = _inline_rescue_test_daemon(tmp_path, monkeypatch)
    daemon._scoped_control_plane_launch = None
    daemon._scoped_recovery_control_plane_launches = {}
    monkeypatch.setattr(
        implementation_daemon,
        "run_process_group_stream",
        lambda command, **_kwargs: subprocess.CompletedProcess(command, 0),
    )
    weakened = SimpleNamespace(
        accepted=True,
        proposal=_test_proposal(
            "proposal:replacement-weakened",
            "RESCUE_MARKER = 'different-proposal'\n",
        ),
    )
    monkeypatch.setattr(
        daemon,
        "_run_validation_with_candidate_binding",
        lambda *_args, **_kwargs: {
            "attempted": True,
            # Model the pre-fix hole: baseline tests and the ordinary proposal
            # gate are green even though the rejected proposal's new security
            # test disappeared.
            "passed": True,
            "returncode": 0,
            "proposal_validation": weakened,
            "proposal_gate": {
                "accepted": True,
                "proposal_id": "proposal:replacement-weakened",
                "receipt_id": "receipt:replacement-weakened",
            },
        },
    )
    task = PortalTask(
        task_id="RESCUE-SECRET-003",
        title="repair a redaction test canary",
        status="in_progress",
        completion="proposal and tests pass",
        priority="high",
        track="test",
        outputs=["tests/unit/test_redaction.py"],
        validation=["python -m pytest -q tests/unit/test_redaction.py"],
    )

    result = daemon._automatic_implementation_rescue(
        task=task,
        attempt=1,
        workspace_path=tmp_path,
        branch_name="agent/rescue-secret-003",
        baseline_ref="a" * 40,
        validation_result=_scoped_test_secret_failure_result(),
        log_path=tmp_path / "implementation.log",
        state=None,
        command=["/opt/providers/grok"],
        base_prompt="Implement the redaction trace test.",
    )

    assert result["passed"] is False
    assert result["auto_rescue_terminal"] is True
    assert result["reason"] == (
        "scoped_test_secret_remediation_test_semantics_not_preserved"
    )
    binding = result["scoped_test_secret_remediation_binding"]
    assert binding["replacement_is_fresh"] is True
    assert binding["replacement_gate_accepted"] is True
    assert binding["full_validation_passed"] is True
    assert binding["test_semantics_preserved"] is False
    assert binding["test_semantics_reason"] == (
        "replacement_inventory_unavailable"
    )
