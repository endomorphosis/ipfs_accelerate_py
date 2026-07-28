"""Tests for deterministic implementation failure review."""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.implementation_failure_review import (
    FAILURE_REVIEW_SCHEMA,
    FailureReviewDecision,
    FailureReviewReason,
    ImplementationFailureReviewReceipt,
    compact_failure_review,
    review_implementation_failure,
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
                "next_attempt_prompt_addendum": "Still required outputs: b.py.",
            },
            "next_attempt_prompt_addendum": "Still required outputs: b.py.",
            "validation_result": {
                "passed": False,
                "returncode": 78,
                "reason": "proposal_gate_failed",
                "failure_review": {
                    "decision": "guide_rescue",
                    "reason_codes": ["incomplete_expected_outputs"],
                },
            },
        }
    )
    assert normalized["failure_review"]["decision"] == "guide_rescue"
    assert "b.py" in normalized["next_attempt_prompt_addendum"]
    assert normalized["validation"]["failure_review"]["decision"] == (
        "guide_rescue"
    )


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


def test_implementation_prompt_policy_appendix_includes_admission_budgets() -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        PortalImplementationDaemon,
        PortalTask,
    )

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
    assert "1000000" in appendix
    assert "tests/fixtures/logic/admissibility" in appendix
    assert "compact recipes" in appendix


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
