"""Tests for deterministic implementation failure review."""

from __future__ import annotations

import pytest
from ipfs_accelerate_py.agent_supervisor.implementation_failure_review import (
    FAILURE_REVIEW_SCHEMA,
    FailureReviewDecision,
    FailureReviewReason,
    ImplementationFailureReviewReceipt,
    compact_failure_review,
    review_implementation_failure,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    canonical_json,
)


def test_guide_rescue_for_incomplete_expected_outputs() -> None:
    review = review_implementation_failure(
        task_id="EVAL-006",
        attempt=2,
        expected_outputs=(
            "benchmarks/semantic_roundtrip/constructors/modal_spacy.py",
            "benchmarks/semantic_roundtrip/stage_metrics.py",
            "tests/unit/benchmarks/semantic_roundtrip/test_modal_spacy_constructor.py",
            "tests/unit/benchmarks/semantic_roundtrip/test_stage_metrics.py",
        ),
        changed_paths=(
            "benchmarks/semantic_roundtrip/constructors/modal_spacy.py",
        ),
        validation_result={
            "attempted": True,
            "passed": False,
            "returncode": 1,
            "reason": "validation_failed",
            "failed_commands": [
                "PYTHONPATH=. python -m pytest "
                "tests/unit/benchmarks/semantic_roundtrip/test_stage_metrics.py -q"
            ],
        },
    )

    assert review.decision is FailureReviewDecision.GUIDE_RESCUE
    assert (
        FailureReviewReason.INCOMPLETE_EXPECTED_OUTPUTS.value
        in review.reason_codes
    )
    assert "stage_metrics.py" in "\n".join(review.missing_expected_outputs)
    assert "declared Outputs" in review.guidance_markdown
    assert "stage_metrics.py" in review.next_attempt_prompt_addendum
    restored = ImplementationFailureReviewReceipt.from_dict(review.to_record())
    assert restored == review
    assert restored.receipt_id == review.receipt_id
    compact = compact_failure_review(review)
    assert compact["decision"] == "guide_rescue"
    assert compact["receipt_id"] == review.receipt_id


def test_directory_output_is_satisfied_by_changed_descendants() -> None:
    review = review_implementation_failure(
        task_id="WALPROC-029",
        attempt=1,
        expected_outputs=(
            "ipfs_datasets_py/ipfs_datasets_py/mcp_server/tools/wallet_processor_tools",
            "ipfs_datasets_py/tests/mcp/test_wallet_processor_tools.py",
        ),
        changed_paths=(
            "ipfs_datasets_py/ipfs_datasets_py/mcp_server/tools/wallet_processor_tools/__init__.py",
            "ipfs_datasets_py/ipfs_datasets_py/mcp_server/tools/wallet_processor_tools/ingest.py",
            "ipfs_datasets_py/tests/mcp/test_wallet_processor_tools.py",
        ),
        validation_result={
            "attempted": True,
            "passed": False,
            "returncode": 1,
            "reason": "validation_failed",
            "failed_commands": [
                "python -m pytest "
                "ipfs_datasets_py/tests/mcp/test_wallet_processor_tools.py -q"
            ],
        },
    )

    assert review.missing_expected_outputs == ()
    assert (
        FailureReviewReason.INCOMPLETE_EXPECTED_OUTPUTS.value
        not in review.reason_codes
    )


def test_directory_output_does_not_admit_prefix_siblings() -> None:
    review = review_implementation_failure(
        task_id="WALPROC-029",
        attempt=1,
        expected_outputs=(
            "ipfs_datasets_py/ipfs_datasets_py/mcp_server/tools/wallet_processor_tools",
        ),
        changed_paths=(
            "ipfs_datasets_py/ipfs_datasets_py/mcp_server/tools/wallet_processor_tools_extra.py",
        ),
        validation_result={
            "attempted": False,
            "passed": False,
            "returncode": 78,
            "reason": "proposal_gate_failed",
        },
    )

    assert review.missing_expected_outputs == (
        "ipfs_datasets_py/ipfs_datasets_py/mcp_server/tools/wallet_processor_tools",
    )
    assert (
        FailureReviewReason.INCOMPLETE_EXPECTED_OUTPUTS.value
        in review.reason_codes
    )


def test_guide_rescue_for_out_of_scope_refactor_paths() -> None:
    review = review_implementation_failure(
        task_id="EVAL-002",
        attempt=1,
        expected_outputs=(
            "benchmarks/semantic_roundtrip_capabilities.py",
            "tests/unit/benchmarks/test_semantic_roundtrip_capabilities.py",
        ),
        changed_paths=(
            "benchmarks/semantic_roundtrip_capabilities.py",
            "benchmarks/semantic_roundtrip/helpers/new_utils.py",
        ),
        validation_result={
            "attempted": False,
            "passed": False,
            "returncode": 78,
            "reason": "proposal_gate_failed",
            "error": "proposal_validation_failed",
            "proposal_gate": {
                "reason_codes": ["path_outside_scope"],
                "changed_paths": [
                    "benchmarks/semantic_roundtrip_capabilities.py",
                    "benchmarks/semantic_roundtrip/helpers/new_utils.py",
                ],
            },
            "scope_adjudication": {
                "accepted": False,
                "justified_paths": [],
                "denied_paths": [
                    "benchmarks/semantic_roundtrip/helpers/new_utils.py"
                ],
            },
        },
    )

    assert review.decision is FailureReviewDecision.GUIDE_RESCUE
    assert (
        FailureReviewReason.SCOPE_EXPANSION_DENIED.value in review.reason_codes
    )
    assert (
        FailureReviewReason.LARGE_OR_UNDECLARED_REFACTOR.value
        in review.reason_codes
    )
    assert "new_utils.py" in review.guidance_markdown
    assert "Do not modify these out-of-scope paths" in (
        review.next_attempt_prompt_addendum
    )


def test_unverifiable_validation_companion_requests_contract_revision() -> None:
    fixture_path = (
        "wallet_interface/ui/tests/fixtures/world-id-fixtures.ts"
    )
    panel_path = (
        "wallet_interface/ui/src/shared/components/"
        "WorldIdVerificationPanel.tsx"
    )
    api_path = (
        "wallet_interface/ui/src/features/wallet/lib/walletApi.ts"
    )
    review = review_implementation_failure(
        task_id="WALPROC-065",
        attempt=1,
        expected_outputs=(api_path, panel_path),
        changed_paths=(api_path, panel_path, fixture_path),
        validation_commands=(
            "npm --prefix wallet_interface/ui test -- --runInBand",
        ),
        proposal_accepted=False,
        scope_adjudication={
            "accepted": False,
            "justified_paths": [],
            "denied_paths": [fixture_path],
            "decisions": [
                {
                    "path": fixture_path,
                    "verdict": "denied",
                    "reason_codes": ["test_change_unverifiable"],
                }
            ],
        },
        validation_result={
            "attempted": False,
            "passed": False,
            "returncode": 78,
            "reason": "proposal_gate_failed",
            "error": "proposal_validation_failed",
            "proposal_gate": {
                "reason_codes": ["path_outside_scope"],
                "changed_paths": [api_path, panel_path, fixture_path],
            },
        },
    )

    assert review.decision is FailureReviewDecision.GUIDE_RESCUE
    assert review.contract_gap_paths == (fixture_path,)
    assert (
        FailureReviewReason.TASK_SCOPE_CONTRACT_REVISION_REQUIRED.value
        in review.reason_codes
    )
    assert "Task-scope contract revision required" in review.guidance_markdown
    assert "protected-board authority" in review.guidance_markdown
    assert fixture_path in review.next_attempt_prompt_addendum
    assert "Do not modify these out-of-scope paths" not in (
        review.next_attempt_prompt_addendum
    )
    restored = ImplementationFailureReviewReceipt.from_dict(
        review.to_record()
    )
    assert restored == review
    assert compact_failure_review(review)["contract_gap_paths"] == [
        fixture_path
    ]


def test_child_repository_validation_target_requests_contract_revision() -> None:
    child_test = "ipfs_kit_py/tests/test_mcp_vfs_adapter_contract.py"
    wrong_root_duplicate = "tests/test_mcp_vfs_adapter_contract.py"
    adapter_path = "ipfs_kit_py/ipfs_kit_py/core/vfs/adapters.py"
    review = review_implementation_failure(
        task_id="KITA-007",
        attempt=1,
        expected_outputs=(adapter_path,),
        changed_paths=(
            adapter_path,
            child_test,
            wrong_root_duplicate,
        ),
        validation_commands=(
            "cd ipfs_kit_py && python -m pytest -q "
            "tests/test_mcp_vfs_adapter_contract.py",
        ),
        proposal_accepted=False,
        scope_adjudication={
            "accepted": False,
            "justified_paths": [],
            "denied_paths": [child_test, wrong_root_duplicate],
            "decisions": [
                {
                    "path": child_test,
                    "verdict": "denied",
                    "reason_codes": ["test_change_unverifiable"],
                },
                {
                    "path": wrong_root_duplicate,
                    "verdict": "denied",
                    "reason_codes": ["test_change_unverifiable"],
                },
            ],
        },
        validation_result={
            "attempted": False,
            "passed": False,
            "returncode": 78,
            "reason": "proposal_gate_failed",
            "error": "proposal_validation_failed",
            "proposal_gate": {
                "reason_codes": ["path_outside_scope"],
                "changed_paths": [
                    adapter_path,
                    child_test,
                    wrong_root_duplicate,
                ],
            },
        },
    )

    assert review.contract_gap_paths == (child_test,)
    assert (
        FailureReviewReason.TASK_SCOPE_CONTRACT_REVISION_REQUIRED.value
        in review.reason_codes
    )
    assert child_test in review.next_attempt_prompt_addendum
    assert wrong_root_duplicate not in review.contract_gap_paths


def test_failed_external_validation_target_requests_contract_revision() -> None:
    owned_test = (
        "ipfs_kit_py/tests/runtime_readiness/mcplusplus/"
        "test_transport_security_parity.py"
    )
    report = (
        "ipfs_kit_py/docs/runtime_readiness/"
        "mcplusplus_conformance.json"
    )
    external_test = (
        "ipfs_kit_py/ipfs_kit_py/mcp_server/tests_e2e_interop.py"
    )
    # Pytest reports this path relative to ``cd ipfs_kit_py``. Its first
    # component happens to equal the child repository name, so the reviewer
    # must bind it to the declared impact identity rather than drop the outer
    # repository prefix.
    runner_reported_test = "ipfs_kit_py/mcp_server/tests_e2e_interop.py"
    command = (
        "cd ipfs_kit_py && python -m pytest -q "
        "tests/runtime_readiness/mcplusplus/"
        "test_transport_security_parity.py "
        "ipfs_kit_py/mcp_server/tests_e2e_interop.py"
    )
    review = review_implementation_failure(
        task_id="KITA-033",
        attempt=1,
        expected_outputs=(owned_test, report),
        validation_commands=(command,),
        validation_result={
            "attempted": True,
            "passed": False,
            "returncode": 1,
            "reason": "declared_validation_failed",
            "proposal_gate": {
                "accepted": True,
                "changed_paths": [owned_test, report],
            },
            "failed_commands": [command],
            "failed_tests": [
                runner_reported_test + "::test_python_import_surface"
            ],
            "failed_test_paths": [runner_reported_test],
            "validation_impact_paths": [owned_test, external_test],
        },
    )

    assert review.contract_gap_paths == (external_test,)
    assert review.out_of_scope_paths == ()
    assert review.justified_paths == ()
    assert (
        FailureReviewReason.TASK_SCOPE_CONTRACT_REVISION_REQUIRED.value
        in review.reason_codes
    )
    assert "routes the repair to their owning task" in review.guidance_markdown
    assert external_test in review.next_attempt_prompt_addendum
    assert runner_reported_test not in review.contract_gap_paths


def test_external_command_target_is_not_gap_without_actual_external_failure() -> None:
    owned_test = "tests/runtime_readiness/test_joined_contract.py"
    external_test = "tests/e2e/test_existing_contract.py"
    command = (
        "python -m pytest -q "
        f"{owned_test} {external_test}"
    )
    review = review_implementation_failure(
        task_id="JOINED-001",
        attempt=1,
        expected_outputs=(owned_test,),
        validation_commands=(command,),
        validation_result={
            "attempted": True,
            "passed": False,
            "returncode": 1,
            "reason": "declared_validation_failed",
            "proposal_gate": {
                "accepted": True,
                "changed_paths": [owned_test],
            },
            "failed_commands": [command],
            "failed_test_paths": [owned_test],
            "validation_impact_paths": [owned_test, external_test],
        },
    )

    assert review.contract_gap_paths == ()
    assert (
        FailureReviewReason.TASK_SCOPE_CONTRACT_REVISION_REQUIRED.value
        not in review.reason_codes
    )


def test_failed_path_requires_declared_validation_impact_binding() -> None:
    owned_test = "tests/runtime_readiness/test_joined_contract.py"
    external_test = "tests/e2e/test_unrelated_contract.py"
    review = review_implementation_failure(
        task_id="JOINED-002",
        attempt=1,
        expected_outputs=(owned_test,),
        validation_result={
            "attempted": True,
            "passed": False,
            "returncode": 1,
            "reason": "declared_validation_failed",
            "proposal_gate": {
                "accepted": True,
                "changed_paths": [owned_test],
            },
            "failed_commands": [
                "python -m pytest -q tests/runtime_readiness"
            ],
            "failed_test_paths": [external_test],
            "validation_impact_paths": [owned_test],
        },
    )

    assert review.contract_gap_paths == ()


def test_validation_selection_impact_paths_are_not_candidate_changes() -> None:
    expected_outputs = (
        "data/validation/conformance-report.json",
        "tests/contract/wallets/test_all_processors.py",
        "tests/contract/wallets/test_worldcoin_differential.py",
    )
    review = review_implementation_failure(
        task_id="WALPROC-027",
        attempt=1,
        expected_outputs=expected_outputs,
        validation_result={
            "attempted": True,
            "passed": False,
            "returncode": 1,
            "reason": "declared_validation_failed",
            "proposal_gate": {
                "accepted": True,
                "changed_paths": list(expected_outputs),
            },
            "selection": {
                "changed_files": [
                    *expected_outputs,
                    "tests/contract/wallets",
                    "tests/unit/wallets",
                ],
            },
            "failed_commands": [
                "python -m pytest -q tests/unit/wallets "
                "tests/contract/wallets"
            ],
        },
    )

    assert review.changed_paths == expected_outputs
    assert review.out_of_scope_paths == ()
    assert (
        FailureReviewReason.SCOPE_EXPANSION_DENIED.value
        not in review.reason_codes
    )
    assert (
        FailureReviewReason.LARGE_OR_UNDECLARED_REFACTOR.value
        not in review.reason_codes
    )
    assert (
        FailureReviewReason.VALIDATION_COMMAND_FAILED.value
        in review.reason_codes
    )


def test_validation_selection_paths_remain_legacy_fallback() -> None:
    review = review_implementation_failure(
        task_id="LEGACY-001",
        attempt=1,
        expected_outputs=("src/runtime.py",),
        validation_result={
            "attempted": True,
            "passed": False,
            "returncode": 1,
            "reason": "declared_validation_failed",
            "selection": {"changed_files": ["src/runtime.py"]},
            "failed_commands": ["python -m pytest -q tests/test_runtime.py"],
        },
    )

    assert review.changed_paths == ("src/runtime.py",)
    assert review.out_of_scope_paths == ()


def test_reject_hard_deny_secret_findings() -> None:
    review = review_implementation_failure(
        task_id="ASI-1",
        attempt=1,
        expected_outputs=("pkg/module.py",),
        changed_paths=("pkg/module.py",),
        validation_result={
            "passed": False,
            "returncode": 78,
            "reason": "proposal_gate_failed",
            "error": "proposal_validation_failed",
            "proposal_gate": {
                "reason_codes": ["secret_change_forbidden"],
            },
        },
    )

    assert review.decision is FailureReviewDecision.REJECT
    assert FailureReviewReason.HARD_DENY_FINDINGS.value in review.reason_codes
    assert "cannot be accepted" in review.guidance_markdown


def test_accept_justified_scope_expansion_when_proposal_gate_only() -> None:
    review = review_implementation_failure(
        task_id="EVAL-005",
        attempt=2,
        expected_outputs=(
            "benchmarks/semantic_roundtrip/selective_repair.py",
            "tests/unit/benchmarks/semantic_roundtrip/test_selective_repair.py",
        ),
        changed_paths=(
            "benchmarks/semantic_roundtrip/selective_repair.py",
            "tests/unit/benchmarks/semantic_roundtrip/test_selective_repair.py",
            "benchmarks/semantic_roundtrip/constructors/typed_deontic.py",
        ),
        proposal_accepted=False,
        scope_adjudication={
            "accepted": True,
            "justified_paths": [
                "benchmarks/semantic_roundtrip/constructors/typed_deontic.py"
            ],
            "denied_paths": [],
        },
        validation_result={
            "passed": False,
            "returncode": 78,
            "reason": "proposal_gate_failed",
            "error": "proposal_validation_failed",
            "proposal_gate": {
                "reason_codes": ["path_outside_scope"],
                "changed_paths": [
                    "benchmarks/semantic_roundtrip/selective_repair.py",
                    "tests/unit/benchmarks/semantic_roundtrip/test_selective_repair.py",
                    "benchmarks/semantic_roundtrip/constructors/typed_deontic.py",
                ],
            },
        },
    )

    # typed_deontic is listed as expected in real EVAL-005 boards; here it is an
    # extra companion. missing_expected is empty because both declared outputs
    # were changed.
    assert review.decision is FailureReviewDecision.ACCEPT
    assert review.accepted is True
    assert (
        FailureReviewReason.SCOPE_EXPANSION_JUSTIFIED.value
        in review.reason_codes
    )
    assert (
        "benchmarks/semantic_roundtrip/constructors/typed_deontic.py"
        in review.justified_paths
    )


def test_environment_failure_is_guided_not_accepted() -> None:
    review = review_implementation_failure(
        task_id="EVAL-001",
        attempt=1,
        expected_outputs=(
            "benchmarks/semantic_roundtrip/evaluation_status.py",
            "tests/unit/benchmarks/semantic_roundtrip/test_evaluation_status.py",
        ),
        changed_paths=(
            "benchmarks/semantic_roundtrip/evaluation_status.py",
            "tests/unit/benchmarks/semantic_roundtrip/test_evaluation_status.py",
        ),
        validation_result={
            "attempted": True,
            "passed": False,
            "returncode": 1,
            "reason": "validation_failed",
            "failed_commands": [
                "PYTHONPATH=. python -m pytest "
                "tests/unit/benchmarks/semantic_roundtrip/test_evaluation_status.py -q"
            ],
        },
        log_excerpt=(
            "/usr/bin/python3.12: No module named pytest\n"
            "[validation failed] returncode=1\n"
        ),
    )

    assert review.decision is FailureReviewDecision.GUIDE_RESCUE
    assert (
        FailureReviewReason.ENVIRONMENT_VALIDATION_UNAVAILABLE.value
        in review.reason_codes
    )
    assert "pytest" in review.guidance_markdown.lower()


def test_receipt_round_trip_and_forged_identity() -> None:
    review = review_implementation_failure(
        task_id="T-1",
        attempt=3,
        expected_outputs=("a.py",),
        changed_paths=("a.py",),
        validation_result={
            "passed": False,
            "returncode": 1,
            "reason": "validation_failed",
        },
    )
    payload = review.to_record()
    assert payload["schema"] == FAILURE_REVIEW_SCHEMA
    assert ImplementationFailureReviewReceipt.from_dict(payload) == review
    forged = dict(payload)
    forged["receipt_id"] = "forged"
    with pytest.raises(ValueError, match="forged"):
        ImplementationFailureReviewReceipt.from_dict(forged)


def test_daemon_normalize_failure_keeps_review_projection() -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        PortalImplementationDaemon,
    )

    normalized = PortalImplementationDaemon._normalize_implementation_failure(
        {
            "kind": "validation_failure",
            "returncode": 78,
            "failure_review": {
                "decision": "guide_rescue",
                "reason_codes": ["incomplete_expected_outputs"],
                "missing_expected_outputs": ["b.py"],
                "contract_gap_paths": ["tests/e2e/test_external.py"],
                "next_attempt_prompt_addendum": "Still required outputs: b.py.",
            },
            "next_attempt_prompt_addendum": "Still required outputs: b.py.",
            "validation_result": {
                "passed": False,
                "returncode": 78,
                "reason": "proposal_gate_failed",
                "failed_tests": [
                    f"tests/test_runtime_{index}.py::test_contract"
                    for index in range(20)
                ],
                "failed_test_paths": [
                    f"tests/test_runtime_{index}.py"
                    for index in range(20)
                ],
                "exception_types": [
                    f"Contract{index}Error"
                    for index in range(12)
                ],
                "validation_impact_paths": [
                    f"tests/test_runtime_{index}.py"
                    for index in range(24)
                ],
                "failure_head": "x" * 3_000,
                "failure_review": {
                    "decision": "guide_rescue",
                    "reason_codes": [
                        "task_scope_contract_revision_required"
                    ],
                    "contract_gap_paths": [
                        "tests/e2e/test_external.py"
                    ],
                },
            },
        }
    )
    assert normalized["failure_review"]["decision"] == "guide_rescue"
    assert "b.py" in normalized["next_attempt_prompt_addendum"]
    assert normalized["failure_review"]["contract_gap_paths"] == [
        "tests/e2e/test_external.py"
    ]
    assert normalized["validation"]["failure_review"]["decision"] == (
        "guide_rescue"
    )
    assert normalized["validation"]["failure_review"][
        "contract_gap_paths"
    ] == ["tests/e2e/test_external.py"]
    assert len(normalized["validation"]["failed_tests"]) == 12
    assert len(normalized["validation"]["failed_test_paths"]) == 12
    assert len(normalized["validation"]["exception_types"]) == 8
    assert len(normalized["validation"]["validation_impact_paths"]) == 16
    assert len(normalized["validation"]["failure_head"]) == 2_000


def test_daemon_retry_context_keeps_contract_gap_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        PortalImplementationDaemon,
        PortalTask,
    )

    daemon = object.__new__(PortalImplementationDaemon)
    captured: dict[str, object] = {}
    sentinel = object()
    monkeypatch.setattr(
        daemon,
        "_implementation_parent",
        lambda _task: (object(), "decision"),
    )
    monkeypatch.setattr(
        daemon,
        "_authoritative_validation_environment_guidance",
        lambda: "sealed validation",
    )
    monkeypatch.setattr(
        daemon,
        "_implementation_checkpoint_manifest",
        lambda _task: {"file_count": 0},
    )

    def capture(
        _task,
        failure,
        *,
        changed_files=(),
        changed_symbols=(),
        unresolved_requirements=(),
    ):
        captured["failure"] = failure
        captured["changed_files"] = changed_files
        captured["changed_symbols"] = changed_symbols
        captured["unresolved_requirements"] = unresolved_requirements
        return sentinel

    monkeypatch.setattr(
        daemon,
        "record_implementation_failure_context",
        capture,
    )
    task = PortalTask(
        task_id="KITA-033",
        title="Joined transport conformance",
        status="todo",
        completion="manual",
        priority="P0",
        track="validation",
        outputs=["tests/runtime_readiness/test_joined.py"],
        validation=[
            "python -m pytest -q tests/runtime_readiness/test_joined.py "
            "tests/e2e/test_external.py"
        ],
    )
    result = daemon._record_failed_attempt_retry_context(
        task,
        returncode=1,
        validation_result={
            "attempted": True,
            "passed": False,
            "returncode": 1,
            "failed_tests": [
                "tests/e2e/test_external.py::test_contract"
            ],
            "failed_test_paths": ["tests/e2e/test_external.py"],
            "validation_impact_paths": [
                "tests/runtime_readiness/test_joined.py",
                "tests/e2e/test_external.py",
            ],
            "failure_head": "E   external contract failed",
            "failure_review": {
                "decision": "guide_rescue",
                "reason_codes": [
                    "task_scope_contract_revision_required"
                ],
                "contract_gap_paths": [
                    "tests/e2e/test_external.py"
                ],
            },
        },
    )

    assert result is sentinel
    failure = captured["failure"]
    assert isinstance(failure, dict)
    assert failure["failure_review"]["contract_gap_paths"] == [
        "tests/e2e/test_external.py"
    ]
    normalized = daemon._normalize_implementation_failure(failure)
    assert normalized["validation"]["failed_tests"] == [
        "tests/e2e/test_external.py::test_contract"
    ]
    assert normalized["validation"]["failed_test_paths"] == [
        "tests/e2e/test_external.py"
    ]
    assert normalized["validation"]["validation_impact_paths"][-1] == (
        "tests/e2e/test_external.py"
    )
    assert normalized["validation"]["failure_head"] == (
        "E   external contract failed"
    )


def test_daemon_normalize_oversized_review_bounds_repeated_guidance() -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        PortalImplementationDaemon,
    )

    guidance = (
        "BEGIN: create src/prover.py and rerun the focused proof test. "
        + ("diagnostic context " * 700)
        + " END: keep tests/prover/test_goal.py green."
    )
    review = {
        "receipt_id": "bafy-reviewed-failure",
        "decision": "guide_rescue",
        "accepted": False,
        "reason_codes": ["incomplete_expected_outputs"],
        "finding_codes": ["path_outside_scope"],
        "missing_expected_outputs": ["src/prover.py"],
        "denied_paths": ["src/undeclared_helper.py"],
        "failed_commands": [
            "python -m pytest tests/prover/test_goal.py -q"
        ],
        "next_attempt_prompt_addendum": guidance,
        "policy_version": "deterministic-failure-review-v2",
    }
    normalized = PortalImplementationDaemon._normalize_implementation_failure(
        {
            "kind": "validation_failure",
            "returncode": 78,
            "failure_review": review,
            "next_attempt_prompt_addendum": guidance,
            "validation_environment_guidance": "sandbox contract " * 500,
            "validation_result": {
                "passed": False,
                "returncode": 78,
                "reason": "proposal_gate_failed",
                "failed_commands": review["failed_commands"],
                "failure_review": review,
            },
        }
    )

    assert len(canonical_json(normalized).encode("utf-8")) <= 16_384
    assert normalized["failure_review"]["receipt_id"] == (
        "bafy-reviewed-failure"
    )
    assert normalized["failure_review"]["decision"] == "guide_rescue"
    assert normalized["failure_review"]["reason_codes"] == [
        "incomplete_expected_outputs"
    ]
    assert normalized["failure_review"]["missing_expected_outputs"] == [
        "src/prover.py"
    ]
    assert "test_goal.py" in normalized["failure_review"]["failed_commands"][0]
    assert "BEGIN: create src/prover.py" in normalized[
        "next_attempt_prompt_addendum"
    ]
    assert "END: keep tests/prover/test_goal.py green" in normalized[
        "next_attempt_prompt_addendum"
    ]
    assert normalized["normalization"]["source_bytes"] > 16_384
    assert normalized["normalization"]["truncated_field_count"] > 0


def test_daemon_normalize_bounds_json_escapes_and_is_canonical() -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        PortalImplementationDaemon,
    )

    guidance = "BEGIN retry\n" + ("\x00" * 20_000) + "\nEND retry"
    forward_kind = {
        "alpha": "a" * 10_000,
        "omega": "z" * 10_000,
    }
    reverse_kind = dict(reversed(tuple(forward_kind.items())))

    def normalize(kind: dict[str, str]) -> dict[str, object]:
        return PortalImplementationDaemon._normalize_implementation_failure(
            {
                "kind": kind,
                "reason": "reviewed validation failure",
                "returncode": 10**3_000,
                "failure_review": {
                    "receipt_id": "bafy-control-review",
                    "decision": "guide_rescue",
                    "reason_codes": ["validation_command_failed"],
                    "missing_expected_outputs": ["src/prover.py"],
                    "failed_commands": [
                        "python -m pytest tests/prover/test_goal.py -q"
                    ],
                    "next_attempt_prompt_addendum": guidance,
                },
                "next_attempt_prompt_addendum": guidance,
            }
        )

    forward = normalize(forward_kind)
    reverse = normalize(reverse_kind)

    assert forward == reverse
    assert len(canonical_json(forward).encode("utf-8")) <= 16_384
    assert forward["normalization"]["source_failure_id"] == reverse[
        "normalization"
    ]["source_failure_id"]
    assert "BEGIN retry" in forward["next_attempt_prompt_addendum"]
    assert "END retry" in forward["next_attempt_prompt_addendum"]
    assert forward["failure_review"]["decision"] == "guide_rescue"
    assert forward["failure_review"]["missing_expected_outputs"] == [
        "src/prover.py"
    ]


def test_directory_outputs_satisfied_by_descendant_changes() -> None:
    """Directory Outputs: are produced when any descendant path changes.

    LIG-016 style tasks declare ``tests/fixtures/...`` as a directory output.
    Filling that tree must not report incomplete_expected_outputs or treat the
    many in-scope fixture files as a large undeclared refactor.
    """

    review = review_implementation_failure(
        task_id="LIG-016",
        attempt=2,
        expected_outputs=(
            "tests/fixtures/logic/admissibility",
            "tests/integration/logic/test_intent_admissibility_gate.py",
        ),
        changed_paths=(
            "tests/fixtures/logic/admissibility/manifest.json",
            "tests/fixtures/logic/admissibility/cases/benign_skill/case.json",
            "tests/fixtures/logic/admissibility/cases/benign_skill/lineage.json",
            "tests/fixtures/logic/admissibility/cases/legal_hard_reject/case.json",
            "tests/integration/logic/test_intent_admissibility_gate.py",
        ),
        validation_result={
            "passed": False,
            "returncode": 78,
            "reason": "proposal_gate_failed",
            "error": "proposal_validation_failed",
            "proposal_gate": {
                "reason_codes": [
                    "output_too_large",
                    "patch_too_large",
                    "patch_parse_error",
                ],
                "changed_paths": [
                    "tests/fixtures/logic/admissibility/manifest.json",
                    "tests/fixtures/logic/admissibility/cases/benign_skill/case.json",
                    "tests/fixtures/logic/admissibility/cases/benign_skill/lineage.json",
                    "tests/fixtures/logic/admissibility/cases/legal_hard_reject/case.json",
                    "tests/integration/logic/test_intent_admissibility_gate.py",
                ],
            },
        },
        proposal_accepted=False,
    )

    assert review.decision is FailureReviewDecision.GUIDE_RESCUE
    assert review.missing_expected_outputs == ()
    assert (
        FailureReviewReason.INCOMPLETE_EXPECTED_OUTPUTS.value
        not in review.reason_codes
    )
    assert (
        FailureReviewReason.LARGE_OR_UNDECLARED_REFACTOR.value
        not in review.reason_codes
    )
    assert (
        FailureReviewReason.PROPOSAL_GATE_FAILED.value in review.reason_codes
    )
    assert "output_too_large" in review.finding_codes
    assert "patch_too_large" in review.finding_codes
    assert "Proposal size" in review.guidance_markdown
    assert "compact" in review.guidance_markdown.lower()
    assert "2_000_000" in review.next_attempt_prompt_addendum.replace(",", "") or (
        "2000000" in review.next_attempt_prompt_addendum
    )
    assert "recipe/generator" in review.next_attempt_prompt_addendum
    assert "Still required outputs" not in review.next_attempt_prompt_addendum


def test_directory_output_still_missing_without_descendant_changes() -> None:
    review = review_implementation_failure(
        task_id="LIG-016",
        attempt=1,
        expected_outputs=(
            "tests/fixtures/logic/admissibility",
            "tests/integration/logic/test_intent_admissibility_gate.py",
        ),
        changed_paths=(
            "tests/integration/logic/test_intent_admissibility_gate.py",
        ),
        validation_result={
            "passed": False,
            "returncode": 1,
            "reason": "validation_failed",
        },
    )

    assert (
        FailureReviewReason.INCOMPLETE_EXPECTED_OUTPUTS.value
        in review.reason_codes
    )
    assert "tests/fixtures/logic/admissibility" in review.missing_expected_outputs
    assert "Still required outputs" in review.next_attempt_prompt_addendum


def test_implementation_prompt_policy_appendix_includes_admission_budgets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        DEFAULT_IMPLEMENTATION_PROPOSAL_FILE_BYTES,
        PortalImplementationDaemon,
        PortalTask,
    )
    from ipfs_accelerate_py.agent_supervisor.validation.validation_runtime import (
        VALIDATION_PATH_ENV,
        canonical_validation_environment_contract,
    )

    monkeypatch.setenv(
        "PATH",
        "/home/test/.elan/bin:/home/test/.local/theorem-provers/bin",
    )
    monkeypatch.delenv(VALIDATION_PATH_ENV, raising=False)
    task = PortalTask(
        task_id="LIG-016",
        title="Integration test for end-to-end admissibility",
        status="pending",
        completion="manual",
        priority="P0",
        track="gate",
        outputs=[
            "tests/fixtures/logic/admissibility",
            "tests/integration/logic/test_intent_admissibility_gate.py",
        ],
        validation=[
            "python -m pytest tests/integration/logic/test_intent_admissibility_gate.py -q"
        ],
        acceptance="Full lineage CIDs asserted; no network required.",
    )
    daemon = object.__new__(PortalImplementationDaemon)
    appendix = PortalImplementationDaemon._implementation_prompt_policy_appendix(
        daemon, task
    )
    assert "Admission policy" in appendix
    assert "directory trees" in appendix
    assert "2000000" in appendix
    assert "2500000" in appendix
    assert str(DEFAULT_IMPLEMENTATION_PROPOSAL_FILE_BYTES) in appendix
    assert "tests/fixtures/logic/admissibility" in appendix
    assert "compact recipes" in appendix
    contract = canonical_validation_environment_contract()
    assert (
        f'`PATH` is exactly "{contract["path"]}"'
        in appendix
    )
    assert "/home/test/.elan/bin" not in appendix
    assert "inherited `PATH` is ignored" in appendix
    assert "ipfs-accelerate-validation-home-" in appendix
    assert "`$HOME/.cache`" in appendix
    assert "`~/.elan`" in appendix
    assert "user-writable tool directories are rejected" in appendix
    assert "never claim usability or weaken mandatory tests" in appendix


def test_failure_review_binds_authoritative_validation_environment() -> None:
    environment_guidance = (
        "## Authoritative validation environment (fail-closed)\n"
        '- `PATH` is exactly "/usr/bin:/bin"; inherited `PATH` is ignored.\n'
        "- `HOME` is private and ephemeral; `XDG_CONFIG_HOME` is under it."
    )
    review = review_implementation_failure(
        task_id="FVT-053",
        attempt=2,
        expected_outputs=("formal_verification_toolchain_certificate.json",),
        changed_paths=("formal_verification_toolchain_certificate.json",),
        validation_result={
            "attempted": True,
            "passed": False,
            "returncode": 1,
            "reason": "declared_validation_failed",
            "failed_commands": [
                "python -m pytest tests/integration/logic/test_toolchain.py -q"
            ],
        },
        validation_environment_guidance=environment_guidance,
    )

    assert environment_guidance in review.guidance_markdown
    assert "Authoritative validation environment:" in (
        review.next_attempt_prompt_addendum
    )
    assert '"/usr/bin:/bin"' in review.next_attempt_prompt_addendum
    restored = ImplementationFailureReviewReceipt.from_dict(review.to_record())
    assert restored.receipt_id == review.receipt_id


def test_size_guidance_when_only_size_findings() -> None:
    review = review_implementation_failure(
        task_id="LIG-016",
        attempt=1,
        expected_outputs=("tests/fixtures/logic/admissibility/manifest.json",),
        changed_paths=("tests/fixtures/logic/admissibility/manifest.json",),
        validation_result={
            "passed": False,
            "returncode": 78,
            "reason": "proposal_gate_failed",
            "error": "proposal_validation_failed",
            "proposal_gate": {
                "reason_codes": ["large_file_forbidden"],
            },
        },
        proposal_accepted=False,
    )

    assert review.decision is FailureReviewDecision.GUIDE_RESCUE
    assert review.missing_expected_outputs == ()
    assert "large_file_forbidden" in review.finding_codes
    assert "Proposal size / bulk limits" in review.guidance_markdown
    assert "large_file_forbidden" in review.next_attempt_prompt_addendum
