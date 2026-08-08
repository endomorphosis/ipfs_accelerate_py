"""Tests for deterministic implementation failure review."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
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


def test_daemon_normalize_failure_bounds_nested_review_without_losing_counterexample() -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        PortalImplementationDaemon,
    )
    private_output = "credential=UNIQUE_PRIVATE_SENTINEL\n" * 40
    failed_commands = [f"pytest tests/test_{index}.py" for index in range(300)]
    failure = {
        "kind": "validation_failure",
        "returncode": 17,
        "reason": "validation_failed",
        "exception_type": "AssertionError",
        "exception_message": "assertion did not match",
        "phase": "validating",
        "failed_commands": failed_commands,
        "failure_review": {
            "receipt_id": "failure-review:receipt-17",
            "decision": "guide_rescue",
            "accepted": False,
            "guidance_markdown": private_output,
            "next_attempt_prompt_addendum": private_output,
        },
        "next_attempt_prompt_addendum": private_output,
        "timeout_policy": {
            "source": "task_metadata",
            "configured_timeout_seconds": 7200,
        },
        "checkpoint_manifest": {
            "manifest_cid": "checkpoint:cid",
            "file_count": 2,
            "total_size_bytes": 99,
        },
        "validation_result": {
            "attempted": True,
            "passed": False,
            "returncode": 17,
            "reason": "validation_failed",
            "failed_command": "python -m pytest test_daemon_port.py -q",
            "failed_commands": failed_commands,
            "failed_tests": [f"test_retry_{index}" for index in range(300)],
            "failed_test_paths": [
                f"test/api/test_{index}.py" for index in range(300)
            ],
            "exception_types": ["AssertionError"],
            "exception_message": "assertion did not match",
            "failure_head": private_output,
            "output": private_output,
            "failure_review": {
                "receipt_id": "failure-review:receipt-17",
                "decision": "guide_rescue",
                "accepted": False,
                "guidance_markdown": private_output,
            },
            "proposal_gate": {
                "accepted": False,
                "proposal_id": "proposal-17",
                "reason_codes": ["validation_failed"],
            },
            "scope_adjudication": {
                "accepted": False,
                "receipt_id": "scope-17",
                "denied_paths": ["outside.py"],
            },
        },
    }

    normalized = PortalImplementationDaemon._normalize_implementation_failure(failure)

    repeated = PortalImplementationDaemon._normalize_implementation_failure(failure)
    wire = json.dumps(normalized, sort_keys=True, separators=(",", ":"))

    assert normalized == repeated
    assert len(wire.encode()) <= 16 * 1024
    assert "UNIQUE_PRIVATE_SENTINEL" not in wire
    assert normalized["exception_type"] == "AssertionError"
    assert normalized["exception_message"] == "assertion did not match"
    assert normalized["phase"] == "validating"
    assert normalized["failure_review"] == {
        "receipt_id": "failure-review:receipt-17",
        "decision": "guide_rescue",
        "accepted": False,
    }
    validation = normalized["validation"]
    assert validation["attempted"] is True
    assert validation["passed"] is False
    assert validation["returncode"] == 17
    assert validation["reason"] == "validation_failed"
    assert validation["failed_command"] == (
        "python -m pytest test_daemon_port.py -q"
    )
    assert validation["failed_commands"]
    assert validation["failed_tests"]
    assert validation["failed_test_paths"]
    assert validation["exception_types"] == ["AssertionError"]
    assert "sha256=" in validation["failure_head"]
    assert validation["failure_review"]["accepted"] is False
    assert normalized["proposal_gate"]["proposal_id"] == "proposal-17"
    assert normalized["scope_adjudication"]["receipt_id"] == "scope-17"
    assert normalized["timeout_policy"]["source"] == "task_metadata"
    assert normalized["checkpoint_manifest"]["total_size_bytes"] == 99
    assert normalized["deduplication"]["deduplicated_occurrence_count"] > 0

    original_tail = json.dumps(
        failed_commands[3:],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    tail_records = [
        record
        for record in normalized["truncation"]["records"]
        if record.get("omitted_item_count") == len(failed_commands) - 3
        and "validation.failed_commands" in record.get("paths", [])
    ]
    assert len(tail_records) == 1
    assert tail_records[0]["original_bytes"] == len(original_tail)
    assert tail_records[0]["sha256"] == hashlib.sha256(
        original_tail
    ).hexdigest()

    class BadMapping(Mapping):
        def __getitem__(self, key):
            raise KeyError(key)

        def __iter__(self):
            return iter(())

        def __len__(self):
            raise RuntimeError("hostile length")

        def get(self, key, default=None):
            return self

    class BadSequence(Sequence):
        def __getitem__(self, index):
            raise RuntimeError("hostile item")

        def __len__(self):
            raise RuntimeError("hostile length")

    class BadMeta(type):
        def __getattribute__(cls, name):
            if name == "__name__":
                raise RuntimeError("hostile type label")
            return super().__getattribute__(name)

    class BadScalar(metaclass=BadMeta):
        def __str__(self):
            raise RuntimeError("hostile rendering")

    class PlainScalar:
        pass

    for hostile in (
        {"failure_review": BadMapping()},
        {"failed_commands": BadSequence()},
        {
            "validation_result": {
                "attempted": True,
                "passed": False,
                "returncode": 4,
                "reason": BadScalar(),
                "failed_command": "pytest hostile.py",
            }
        },
    ):
        hostile_result = (
            PortalImplementationDaemon._normalize_implementation_failure(
                hostile
            )
        )
        assert len(
            json.dumps(
                hostile_result,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ) <= 16 * 1024

    first_plain = PortalImplementationDaemon._normalize_implementation_failure(
        {"validation_result": {"reason": PlainScalar()}}
    )
    second_plain = PortalImplementationDaemon._normalize_implementation_failure(
        {"validation_result": {"reason": PlainScalar()}}
    )
    assert first_plain == second_plain

    class FlipInt(int):
        def __int__(self):
            raise AssertionError("custom integer conversion must not run")

    class FlipList(list):
        def __iter__(self):
            raise AssertionError("custom sequence iteration must not run")

        def __getitem__(self, index):
            raise AssertionError("custom sequence lookup must not run")

    hooked_failure = {
        "returncode": FlipInt(17),
        "validation_result": {
            "attempted": True,
            "passed": False,
            "returncode": FlipInt(17),
            "failed_tests": FlipList(["test_hook.py::test_hook"]),
        },
    }
    assert PortalImplementationDaemon._normalize_implementation_failure(
        hooked_failure
    ) == PortalImplementationDaemon._normalize_implementation_failure(
        hooked_failure
    )


def test_daemon_normalize_failure_last_resort_is_private_and_bounded(
    monkeypatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        PortalImplementationDaemon,
    )
    original_unchecked = (
        PortalImplementationDaemon._normalize_implementation_failure_unchecked
    )

    def fail_unchecked(_failure):
        raise KeyboardInterrupt("force the exact-container projection")

    monkeypatch.setattr(
        PortalImplementationDaemon,
        "_normalize_implementation_failure_unchecked",
        staticmethod(fail_unchecked),
    )
    secret = "OPAQUE_LAST_RESORT_PRIVATE_SENTINEL"
    long_identifier = "L" * 100_000
    failed_commands = [
        "pytest tests/test_retry.py -q",
        "pytest tests/test_retry_two.py -q",
        "pytest tests/test_retry_three.py --token " + secret,
    ]
    failure = {
        "kind": "validation_failure",
        "returncode": 19,
        "failure_review": {
            "receipt_id": "failure-review:last-resort",
            "decision": "guide_rescue",
            "accepted": False,
            "guidance_markdown": secret,
        },
        "timeout_policy": {
            "source": long_identifier,
            "configured_timeout_seconds": 7200,
        },
        "checkpoint_manifest": {
            "schema": "checkpoint@1",
            "manifest_cid": long_identifier,
            "file_count": 2,
            "total_size_bytes": 99,
        },
        "validation_result": {
            "attempted": True,
            "passed": False,
            "returncode": 19,
            "reason": "validation_failed",
            "failed_command": failed_commands[0],
            "failed_commands": failed_commands,
            "failed_tests": [
                "tests/test_retry.py::test_retry[" + secret + "]",
                "tests/test_retry.py::test_other",
            ],
            "failed_test_paths": ["tests/test_retry.py"],
            "exception_types": ["AssertionError"],
            "exception_message": secret,
            "failure_head": "E AssertionError: " + secret,
            "output": secret + ("-private-body" * 1_000),
            "proposal_gate": {
                "accepted": False,
                "attempted": True,
                "proposal_id": long_identifier,
                "reason_codes": ["scope_denied"],
                "changed_paths": ["a.py"],
            },
            "scope_adjudication": {
                "accepted": False,
                "receipt_id": long_identifier,
                "authorized_paths": ["a.py"],
                "denied_paths": ["b.py"],
            },
        },
    }

    normalized = PortalImplementationDaemon._normalize_implementation_failure(
        failure
    )
    wire = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    assert len(wire.encode()) <= 16 * 1024
    assert secret not in wire
    assert normalized["validation"]["attempted"] is True
    assert normalized["validation"]["passed"] is False
    assert normalized["validation"]["returncode"] == 19
    assert normalized["validation"]["reason"] == "validation_failed"
    assert normalized["validation"]["failed_command"] == failed_commands[0]
    assert "param-sha256=" in normalized["validation"]["failed_tests"][0]
    assert "failure-head-omitted" in normalized["validation"]["failure_head"]
    assert normalized["failure_review"]["receipt_id"] == (
        "failure-review:last-resort"
    )
    assert "truncated original_bytes=100000" in normalized[
        "proposal_gate"
    ]["proposal_id"]
    assert "truncated original_bytes=100000" in normalized[
        "scope_adjudication"
    ]["receipt_id"]
    assert normalized["proposal_gate"]["reason_codes"] == ["scope_denied"]
    assert normalized["proposal_gate"]["changed_paths"] == ["a.py"]
    assert normalized["scope_adjudication"]["authorized_paths"] == [
        "a.py"
    ]
    assert normalized["scope_adjudication"]["denied_paths"] == ["b.py"]
    assert "truncated original_bytes=100000" in normalized[
        "timeout_policy"
    ]["source"]
    assert "truncated original_bytes=100000" in normalized[
        "checkpoint_manifest"
    ]["manifest_cid"]
    expected_tail = json.dumps(
        failed_commands[1:], sort_keys=True, separators=(",", ":")
    ).encode()
    assert any(
        record.get("omitted_item_count") == 2
        and record.get("original_bytes") == len(expected_tail)
        and record.get("sha256")
        == hashlib.sha256(expected_tail).hexdigest()
        for record in normalized["truncation"]["records"]
    )

    def fail_failure_head(_cls, _value):
        raise ValueError("force unchecked helper failure")

    monkeypatch.setattr(
        PortalImplementationDaemon,
        "_normalize_implementation_failure_unchecked",
        staticmethod(original_unchecked),
    )
    monkeypatch.setattr(
        PortalImplementationDaemon,
        "_sanitize_retry_failure_head",
        classmethod(fail_failure_head),
    )
    inner_fallback = (
        PortalImplementationDaemon._normalize_implementation_failure(failure)
    )
    inner_wire = json.dumps(
        inner_fallback, sort_keys=True, separators=(",", ":")
    )
    assert len(inner_wire.encode()) <= 16 * 1024
    assert secret not in inner_wire
    assert inner_fallback["validation"]["attempted"] is True
    assert inner_fallback["validation"]["passed"] is False
    assert inner_fallback["validation"]["returncode"] == 19
    for key in (
        "proposal_gate",
        "scope_adjudication",
        "timeout_policy",
        "checkpoint_manifest",
    ):
        assert key in inner_fallback


def test_daemon_normalize_failure_final_envelope_refreshes_tail_ledger() -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        PortalImplementationDaemon,
    )

    def oversized(prefix: str) -> list[str]:
        return [f"{prefix}-{index}-" + ("x" * 5_000) for index in range(20)]

    review_list_keys = (
        "reason_codes",
        "finding_codes",
        "missing_expected_outputs",
        "out_of_scope_paths",
        "justified_paths",
        "denied_paths",
        "contract_gap_paths",
        "failed_commands",
    )
    review_body_keys = (
        "guidance_markdown",
        "review_markdown",
        "body",
        "analysis",
        "raw_response",
        "next_attempt_prompt_addendum",
    )
    huge = "L" * 5_000

    def review(prefix: str, list_count: int) -> dict[str, object]:
        return {
            "receipt_id": huge,
            "decision": huge,
            "policy_version": huge,
            "accepted": False,
            **{
                key: oversized(prefix + key)
                for key in review_list_keys[:list_count]
            },
            **{key: huge for key in review_body_keys},
        }

    validation_commands = oversized("vc")
    failure = {
        "kind": huge,
        "returncode": 13,
        **{
            key: huge
            for key in (
                "reason",
                "exception_type",
                "exception_message",
                "message",
                "phase",
                "timeout_reason",
                "counterexample_id",
            )
        },
        **{
            key: oversized("source-" + key)
            for key in (
                "reason_codes",
                "failed_commands",
                "failing_checks",
                "missing_outputs",
                "counterexample_ids",
            )
        },
        "failure_review": review("source-review-", 0),
        "next_attempt_prompt_addendum": huge,
        "timeout_policy": {
            "source": huge,
            "configured_timeout_seconds": 2**31 - 1,
            "progress_timeout_seconds": 2**31 - 1,
            "max_timeout_seconds": 2**31 - 1,
            "progress_aware": True,
        },
        "checkpoint_manifest": {
            "schema": huge,
            "manifest_cid": huge,
            "file_count": 2**31 - 1,
            "total_size_bytes": 2**31 - 1,
            "total_bytes": 2**31 - 1,
        },
        "validation_result": {
            "attempted": True,
            "passed": False,
            "returncode": 13,
            **{
                key: huge
                for key in (
                    "reason",
                    "error",
                    "exception_message",
                    "failure_head",
                )
            },
            "failed_command": huge,
            "failed_commands": validation_commands,
            "failed_tests": oversized("test"),
            "failed_test_paths": oversized("path"),
            "exception_types": oversized("exception"),
            "reason_codes": oversized("validation-reason"),
            "failure_review": review("validation-review-", 4),
            "next_attempt_prompt_addendum": huge,
            **{
                key: f"{key}-" + ("o" * 9_000)
                for key in ("output", "stdout", "stderr", "raw_output")
            },
            "proposal_gate": {
                "accepted": False,
                **{
                    key: huge
                    for key in (
                        "proposal_id",
                        "policy_id",
                        "receipt_id",
                        "repository_tree_id",
                    )
                },
                "reason_codes": oversized("proposal-reason"),
                "changed_paths": oversized("proposal-path"),
            },
            "scope_adjudication": {
                "accepted": False,
                "receipt_id": huge,
                "proposal_id": huge,
                "authorized_paths": oversized("authorized"),
                "denied_paths": oversized("denied"),
            },
        },
    }

    normalized = PortalImplementationDaemon._normalize_implementation_failure(
        failure
    )
    repeated = PortalImplementationDaemon._normalize_implementation_failure(
        failure
    )
    wire = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    assert normalized == repeated
    assert len(wire.encode()) <= 16 * 1024
    for key in (
        "proposal_gate",
        "scope_adjudication",
        "timeout_policy",
        "checkpoint_manifest",
    ):
        assert key in normalized

    truncation = normalized["truncation"]
    records = truncation["records"]
    aggregate_count = truncation.get("record_set", {}).get("record_count", 0)
    assert normalized["deduplication"]["unique_omission_count"] == (
        len(records) + aggregate_count
    )

    # The former intermediate-return bug created this second-pass tail marker
    # after snapshotting the omission ledger.  Reconstruct it and require the
    # terminal envelope to retain its explicit hash, byte count, and item count.
    def projected_item(value: str) -> str:
        raw = value.encode()
        digest = hashlib.sha256(raw).hexdigest()
        marker = (
            f"[truncated original_bytes={len(raw)} sha256={digest}]"
        )
        head = raw[: 192 - len(marker.encode()) - 1].decode()
        return head + "\n" + marker

    projected = [projected_item(value) for value in validation_commands[:3]]
    source_tail = json.dumps(
        validation_commands[3:], sort_keys=True, separators=(",", ":")
    ).encode()
    source_tail_digest = hashlib.sha256(source_tail).hexdigest()
    projected.append(
        f"[truncated original_bytes={len(source_tail)} "
        f"sha256={source_tail_digest} omitted_items=17]"
    )
    first_fallback_tail = json.dumps(
        projected[2:], sort_keys=True, separators=(",", ":")
    ).encode()
    first_fallback_marker = (
        f"[truncated original_bytes={len(first_fallback_tail)} "
        f"sha256={hashlib.sha256(first_fallback_tail).hexdigest()}]"
    )
    second_fallback_tail = json.dumps(
        [projected[1], first_fallback_marker],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    second_fallback_digest = hashlib.sha256(second_fallback_tail).hexdigest()
    assert any(
        record.get("original_bytes") == len(second_fallback_tail)
        and record.get("sha256") == second_fallback_digest
        and record.get("omitted_item_count") == 2
        for record in records
    )

    for values in (
        normalized["validation"].get("failed_commands", []),
        normalized["validation"].get("failed_tests", []),
        normalized["validation"].get("failed_test_paths", []),
        normalized["validation"].get("exception_types", []),
    ):
        for value in values:
            if not value.startswith("[truncated original_bytes="):
                continue
            parts = value.rstrip("]").split()
            original_bytes = int(parts[1].split("=", 1)[1])
            digest = parts[2].split("=", 1)[1]
            assert any(
                record.get("original_bytes") == original_bytes
                and record.get("sha256") == digest
                and record.get("omitted_item_count", 0) > 0
                for record in records
            )


def test_daemon_normalize_failure_hides_private_fragments_and_counts_addenda() -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        PortalImplementationDaemon,
    )

    output_prefix = "PRIVATE_OUTPUT_PREFIX_WITHOUT_KEYWORD"
    review_prefix = "PRIVATE_REVIEW_PREFIX_WITHOUT_KEYWORD"
    output_body = output_prefix + ("-output" * 10_000)
    review_body = review_prefix + ("-review" * 10_000)
    sensitive_command = "pytest test_auth.py --api-key OPAQUE_COMMAND_SECRET " + (
        "x" * 5_000
    )
    normalized = PortalImplementationDaemon._normalize_implementation_failure(
        {
            "returncode": 31,
            "failure_review": {
                "receipt_id": "failure-review:private",
                "guidance_markdown": review_body,
            },
            "next_attempt_prompt_addendum": review_prefix,
            "validation_result": {
                "attempted": True,
                "passed": False,
                "returncode": 31,
                "output": output_body,
                "failure_head": output_prefix,
                "failed_command": sensitive_command,
            },
        }
    )
    wire = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    assert output_prefix not in wire
    assert review_prefix not in wire
    assert "OPAQUE_COMMAND_SECRET" not in wire
    assert "--api-key=<redacted sha256=" in wire
    sensitive_bytes = sensitive_command.encode()
    records = normalized["truncation"]["records"]
    assert any(
        record["original_bytes"] == len(sensitive_bytes)
        and record["sha256"] == hashlib.sha256(sensitive_bytes).hexdigest()
        for record in records
    )

    shared_addendum = "PRIVATE_ADDENDUM_BODY_WITHOUT_KEYWORD\n" * 160
    repeated = PortalImplementationDaemon._normalize_implementation_failure(
        {
            "returncode": 32,
            "failure_review": {
                "receipt_id": "failure-review:addendum",
                "next_attempt_prompt_addendum": shared_addendum,
            },
            "next_attempt_prompt_addendum": shared_addendum,
            "validation_result": {
                "attempted": True,
                "passed": False,
                "returncode": 32,
                "next_attempt_prompt_addendum": shared_addendum,
                "failure_review": {
                    "receipt_id": "failure-review:addendum",
                    "next_attempt_prompt_addendum": shared_addendum,
                },
            },
        }
    )
    addendum_bytes = shared_addendum.encode()
    addendum_records = [
        record
        for record in repeated["truncation"]["records"]
        if record["original_bytes"] == len(addendum_bytes)
        and record["sha256"] == hashlib.sha256(addendum_bytes).hexdigest()
    ]
    assert len(addendum_records) == 1
    assert addendum_records[0]["occurrence_count"] >= 4
    assert repeated["deduplication"][
        "deduplicated_occurrence_count"
    ] >= 3

    private_fragment = "OPAQUE_PRIVATE_FRAGMENT_7391"
    bearer_secret = "BEARER_PRIVATE_ABC987"
    contained = PortalImplementationDaemon._normalize_implementation_failure(
        {
            "returncode": 33,
            "failure_review": {
                "receipt_id": "failure-review:contained",
                "guidance_markdown": private_fragment,
            },
            "next_attempt_prompt_addendum": (
                "Repair because " + private_fragment
            ),
            "validation_result": {
                "attempted": True,
                "passed": False,
                "returncode": 33,
                "reason": "validation_failed",
                "failed_command": (
                    "curl -H 'Authorization: Bearer "
                    + bearer_secret
                    + "' --value "
                    + private_fragment
                ),
                "failed_tests": [
                    "tests/test_auth.py::test_login["
                    + private_fragment
                    + "]"
                ],
                "failure_head": (
                    "E AssertionError: expected " + private_fragment
                ),
                "output": private_fragment,
            },
        }
    )
    contained_wire = json.dumps(
        contained, sort_keys=True, separators=(",", ":")
    )
    assert private_fragment not in contained_wire
    assert bearer_secret not in contained_wire
    assert "Authorization=<redacted sha256=" in contained_wire
    assert "param-sha256=" in contained["validation"]["failed_tests"][0]
    assert "failure-head-omitted" in contained["validation"][
        "failure_head"
    ]

    echoed_core = {
        "reason": "declared_validation_failed",
        "failed_command": "pytest tests/test_core.py -q",
        "failed_tests": ["tests/test_core.py::test_core"],
        "failed_test_paths": ["tests/test_core.py"],
        "exception_types": ["AssertionError"],
        "failure_head": "E AssertionError: core failed",
    }
    echoed = PortalImplementationDaemon._normalize_implementation_failure(
        {
            "returncode": 34,
            "validation_result": {
                "attempted": True,
                "passed": False,
                "returncode": 34,
                **echoed_core,
                "output": " ".join(
                    (
                        echoed_core["reason"],
                        echoed_core["failed_command"],
                        echoed_core["failed_tests"][0],
                        echoed_core["failed_test_paths"][0],
                        echoed_core["exception_types"][0],
                        echoed_core["failure_head"],
                    )
                ),
            },
        }
    )
    assert echoed["validation"]["reason"] == echoed_core["reason"]
    assert echoed["validation"]["failed_command"] == echoed_core[
        "failed_command"
    ]
    assert echoed["validation"]["failed_tests"] == echoed_core[
        "failed_tests"
    ]
    assert echoed["validation"]["failed_test_paths"] == echoed_core[
        "failed_test_paths"
    ]
    assert echoed["validation"]["exception_types"] == echoed_core[
        "exception_types"
    ]


def test_daemon_normalize_failure_emergency_projection_keeps_core() -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        PortalImplementationDaemon,
    )

    def oversized(prefix: str) -> list[str]:
        return [f"{prefix}-{index}-" + ("x" * 5_000) for index in range(20)]

    review_lists = {
        key: oversized(f"review-{key}")
        for key in (
            "reason_codes",
            "finding_codes",
            "missing_expected_outputs",
            "out_of_scope_paths",
            "justified_paths",
            "denied_paths",
            "contract_gap_paths",
            "failed_commands",
        )
    }
    failure = {
        "kind": "validation_failure",
        "returncode": 23,
        "reason": "validation_failed",
        "exception_type": "AssertionError",
        "exception_message": "expected retry evidence",
        "phase": "validating",
        "reason_codes": oversized("reason"),
        "failed_commands": oversized("command"),
        "failing_checks": oversized("check"),
        "missing_outputs": oversized("missing"),
        "counterexample_ids": oversized("counterexample"),
        "failure_review": {
            "receipt_id": "failure-review:emergency-23",
            "decision": "guide_rescue",
            "accepted": False,
            **review_lists,
        },
        "timeout_policy": {
            "source": "task_metadata",
            "configured_timeout_seconds": 7200,
        },
        "checkpoint_manifest": {
            "manifest_cid": "checkpoint:emergency",
            "file_count": 3,
            "total_size_bytes": 123,
        },
        "validation_result": {
            "attempted": True,
            "passed": False,
            "returncode": 23,
            "reason": "validation_failed",
            "failed_command": "pytest tests/test_retry.py -q",
            "failed_commands": oversized("validation-command"),
            "failed_tests": oversized("test"),
            "failed_test_paths": oversized("path"),
            "exception_types": oversized("exception"),
            "exception_message": "expected retry evidence",
            "failure_head": "assert retry receipt",
            "failure_review": {
                "receipt_id": "failure-review:emergency-23",
                "decision": "guide_rescue",
                "accepted": False,
            },
            "proposal_gate": {
                "accepted": False,
                "proposal_id": "proposal-emergency",
                "reason_codes": oversized("proposal-reason"),
            },
            "scope_adjudication": {
                "accepted": False,
                "receipt_id": "scope-emergency",
                "authorized_paths": oversized("authorized"),
                "denied_paths": oversized("denied"),
            },
        },
    }

    normalized = PortalImplementationDaemon._normalize_implementation_failure(
        failure
    )
    wire = json.dumps(normalized, sort_keys=True, separators=(",", ":"))

    assert len(wire.encode()) <= 16 * 1024
    assert "normalization_truncation" in normalized
    assert normalized["failure_review"]["receipt_id"] == (
        "failure-review:emergency-23"
    )
    assert normalized["exception_type"] == "AssertionError"
    assert normalized["exception_message"] == "expected retry evidence"
    assert normalized["phase"] == "validating"
    validation = normalized["validation"]
    assert validation["attempted"] is True
    assert validation["passed"] is False
    assert validation["returncode"] == 23
    assert validation["reason"] == "validation_failed"
    assert validation["failed_command"] == "pytest tests/test_retry.py -q"
    assert validation["failed_commands"]
    assert validation["failed_tests"]
    assert validation["failed_test_paths"]
    assert validation["exception_types"]
    assert validation["exception_message"] == "expected retry evidence"
    failure_head_bytes = b"assert retry receipt"
    assert validation["failure_head"].startswith(
        "[failure-head-omitted original_bytes=20 sha256="
    )
    assert hashlib.sha256(failure_head_bytes).hexdigest() in validation[
        "failure_head"
    ]
    assert normalized["proposal_gate"]["proposal_id"] == (
        "proposal-emergency"
    )
    assert normalized["scope_adjudication"]["receipt_id"] == (
        "scope-emergency"
    )
    assert normalized["timeout_policy"]["source"] == "task_metadata"
    assert normalized["checkpoint_manifest"]["total_size_bytes"] == 123
    expected_test_tail = json.dumps(
        failure["validation_result"]["failed_tests"][3:],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    expected_test_tail_sha = hashlib.sha256(expected_test_tail).hexdigest()
    test_tail_records = [
        record
        for record in normalized["truncation"]["records"]
        if record.get("omitted_item_count") == 17
        and record.get("original_bytes") == len(expected_test_tail)
        and record.get("sha256") == expected_test_tail_sha
    ]
    assert len(test_tail_records) == 1
    assert test_tail_records[0]["marker"] == (
        f"[truncated original_bytes={len(expected_test_tail)} "
        f"sha256={expected_test_tail_sha}]"
    )
    truncation = normalized["truncation"]
    represented_omissions = len(truncation["records"]) + int(
        truncation.get("record_set", {}).get("record_count", 0)
    )
    assert normalized["deduplication"]["unique_omission_count"] == (
        represented_omissions
    )
    assert normalized["deduplication"]["occurrence_count"] >= (
        represented_omissions
    )

    class EvilStr(str):
        def encode(self, *args, **kwargs):
            return 42

    malformed = {
        "reason_codes": [EvilStr("optional-malformed")],
        "failure_review": {
            "receipt_id": "failure-review:malformed",
            "decision": "guide_rescue",
        },
        "validation_result": {
            "attempted": True,
            "passed": False,
            "returncode": 29,
            "reason": "validation_failed",
            "failed_command": "pytest tests/test_malformed.py -q",
            "failure_head": "malformed optional reviewer value",
        },
    }
    malformed_result = (
        PortalImplementationDaemon._normalize_implementation_failure(
            malformed
        )
    )
    assert malformed_result["validation"]["attempted"] is True
    assert malformed_result["validation"]["passed"] is False
    assert malformed_result["validation"]["returncode"] == 29
    assert malformed_result["validation"]["reason"] == "validation_failed"
    assert malformed_result["failure_review"]["receipt_id"] == (
        "failure-review:malformed"
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
