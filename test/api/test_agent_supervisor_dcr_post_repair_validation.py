"""DCR-073: validate, reindex, observe, and re-prove the new epoch.

Acceptance:
* Finding disappears for the intended semantic reason.
* No protected invariant regresses.
* All mandatory gates run (skipped/unsupported fail closed).
* Expected results never substitute for detector output.
* Synthetic release children fail closed.
* Zero model/provider calls are observed.
* Transaction must contain actual source edits.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.autonomous_repair.transaction import (
    TransactionDisposition,
)
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.validation import (
    MANDATORY_GATES,
    POST_REPAIR_VALIDATION_INTERFACE,
    PROTECTED_INVARIANT_KEYS,
    REPAIR_PROOF_TRANSITION_INTERFACE,
    CommandGateEvidence,
    FindingClosureEvidence,
    GateDisposition,
    GateResult,
    LiveObservationEvidence,
    PostRepairEpochEvidence,
    PostRepairValidationError,
    PostRepairValidationReason,
    PostRepairValidator,
    ProtectedInvariantEvidence,
    ReindexRecompileEvidence,
    RepairProofTransition,
    TransactionSourceEvidence,
    ValidationDisposition,
    materialize_post_repair_epoch,
    validate_post_repair_epoch,
)


PATH = "external/ipfs_accelerate/ipfs_accelerate_py/sample.py"
FINDING = "finding:dcr073-edge"
SEMANTIC = "edge_resolved_and_obligation_proved"


def _transaction(**changes: object) -> TransactionSourceEvidence:
    values: dict[str, object] = {
        "transaction_id": "tx:dcr073",
        "disposition": TransactionDisposition.COMMITTED.value,
        "changed_paths": (PATH,),
        "before_hashes": {PATH: "sha256:" + ("11" * 32)},
        "after_hashes": {PATH: "sha256:" + ("22" * 32)},
        "root_ids": ("external/ipfs_accelerate",),
        "forest_id": "forest:dcr073",
        "tree_id": "tree:dcr073",
        "candidate_epoch_id": "epoch:dcr073-post",
    }
    values.update(changes)
    return TransactionSourceEvidence(**values)  # type: ignore[arg-type]


def _commands(**changes: object) -> CommandGateEvidence:
    values: dict[str, object] = {
        "format_passed": True,
        "type_passed": True,
        "unit_passed": True,
        "negative_passed": True,
        "command_receipt_ids": ("cmd:format", "cmd:type", "cmd:unit", "cmd:negative"),
    }
    values.update(changes)
    return CommandGateEvidence(**values)  # type: ignore[arg-type]


def _live(**changes: object) -> LiveObservationEvidence:
    values: dict[str, object] = {
        "service_roles": ("accelerate", "datasets"),
        "started_service_ids": ("accelerate:runtime", "datasets:runtime"),
        "tools_list_receipt_ids": ("live:list",),
        "tools_call_receipt_ids": ("live:call",),
        "invalid_call_receipt_ids": ("live:invalid",),
        "observed_tools": ("accelerate.inference",),
        "expected_tools": ("accelerate.inference",),
        "evidence_ids": ("live:obs",),
    }
    values.update(changes)
    return LiveObservationEvidence(**values)  # type: ignore[arg-type]


def _reindex(**changes: object) -> ReindexRecompileEvidence:
    values: dict[str, object] = {
        "index_id": "index:dcr073",
        "reindex_receipt_id": "reindex:dcr073",
        "recompile_receipt_id": "recompile:dcr073",
        "rebuilt_paths": (PATH,),
        "clean_rebuild_equivalent": True,
    }
    values.update(changes)
    return ReindexRecompileEvidence(**values)  # type: ignore[arg-type]


def _finding(**changes: object) -> FindingClosureEvidence:
    values: dict[str, object] = {
        "finding_id": FINDING,
        "closed": True,
        "semantic_reason": SEMANTIC,
        "intended_semantic_reason": SEMANTIC,
        "residual_finding_ids": (),
        "evidence_ids": ("finding:closed",),
    }
    values.update(changes)
    return FindingClosureEvidence(**values)  # type: ignore[arg-type]


def _invariants(**changes: object) -> ProtectedInvariantEvidence:
    values: dict[str, object] = {
        "checked_keys": PROTECTED_INVARIANT_KEYS,
        "regressed_keys": (),
        "evidence_ids": ("invariant:ok",),
    }
    values.update(changes)
    return ProtectedInvariantEvidence(**values)  # type: ignore[arg-type]


def _transition(**changes: object) -> RepairProofTransition:
    values: dict[str, object] = {
        "pre_epoch_id": "epoch:dcr073-pre",
        "post_epoch_id": "epoch:dcr073-post",
        "pre_proof_ids": ("proof:pre",),
        "post_proof_ids": ("proof:post",),
        "closed_finding_ids": (FINDING,),
        "residual_finding_ids": (),
        "reconstructed": True,
        "evidence_ids": ("proof:transition",),
    }
    values.update(changes)
    return RepairProofTransition(**values)  # type: ignore[arg-type]


def complete_evidence(**changes: object) -> PostRepairEpochEvidence:
    values: dict[str, object] = {
        "transaction": _transaction(),
        "commands": _commands(),
        "live": _live(),
        "reindex": _reindex(),
        "finding": _finding(),
        "invariants": _invariants(),
        "proof_transition": _transition(),
        "model_invocation_count": 0,
        "provider_invocation_count": 0,
    }
    values.update(changes)
    return PostRepairEpochEvidence(**values)  # type: ignore[arg-type]


def test_interfaces_exported() -> None:
    assert POST_REPAIR_VALIDATION_INTERFACE == "PostRepairValidation@1"
    assert REPAIR_PROOF_TRANSITION_INTERFACE == "RepairProofTransition@1"
    assert PostRepairValidator.INTERFACE == POST_REPAIR_VALIDATION_INTERFACE
    assert RepairProofTransition.INTERFACE == REPAIR_PROOF_TRANSITION_INTERFACE
    assert set(MANDATORY_GATES) >= {
        "source_edits",
        "format",
        "type",
        "unit",
        "negative",
        "service_start",
        "live_reobserve",
        "reindex",
        "recompile",
        "proof_reconstruction",
        "finding_closed",
        "protected_invariants",
        "zero_model_calls",
    }


def test_passing_epoch_claims_completion_with_all_gates() -> None:
    report = validate_post_repair_epoch(complete_evidence())
    assert report.ok is True
    assert report.disposition is ValidationDisposition.PASSED
    assert report.claims_completion is True
    assert report.runtime_model_calls == 0
    assert report.runtime_provider_calls == 0
    assert report.grants_write_authority is False
    assert {item.gate for item in report.gate_results} == set(MANDATORY_GATES)
    assert all(item.ok for item in report.gate_results)
    assert all(item.ran for item in report.gate_results)
    assert report.proof_transition.ok is True
    assert report.finding_id == FINDING


def test_no_source_edits_fails() -> None:
    evidence = complete_evidence(
        transaction=_transaction(
            before_hashes={PATH: "sha256:" + ("aa" * 32)},
            after_hashes={PATH: "sha256:" + ("aa" * 32)},
        )
    )
    report = validate_post_repair_epoch(evidence)
    assert report.ok is False
    assert report.claims_completion is False
    assert PostRepairValidationReason.NO_SOURCE_EDITS.value in report.reason_codes


def test_uncommitted_transaction_fails() -> None:
    evidence = complete_evidence(
        transaction=_transaction(disposition=TransactionDisposition.OPEN.value)
    )
    report = validate_post_repair_epoch(evidence)
    assert report.ok is False
    assert (
        PostRepairValidationReason.TRANSACTION_NOT_COMMITTED.value
        in report.reason_codes
    )


def test_expected_tools_without_observation_fails() -> None:
    evidence = complete_evidence(
        live=_live(observed_tools=(), tools_list_receipt_ids=("live:list",))
    )
    report = validate_post_repair_epoch(evidence)
    assert report.ok is False
    assert (
        PostRepairValidationReason.EXPECTED_SUBSTITUTED_FOR_DETECTOR.value
        in report.reason_codes
    )


def test_expected_only_command_results_fail() -> None:
    evidence = complete_evidence(commands=_commands(expected_only=True))
    report = validate_post_repair_epoch(evidence)
    assert report.ok is False
    assert (
        PostRepairValidationReason.EXPECTED_SUBSTITUTED_FOR_DETECTOR.value
        in report.reason_codes
    )
    failed = [item for item in report.gate_results if item.gate == "format"][0]
    assert failed.expected_only is True


def test_skipped_mandatory_gate_fails() -> None:
    override = GateResult(
        gate="unit",
        disposition=GateDisposition.SKIPPED,
        detector_id="detector:unit",
        ran=False,
    )
    evidence = complete_evidence(gate_overrides={"unit": override})
    report = validate_post_repair_epoch(evidence)
    assert report.ok is False
    assert (
        PostRepairValidationReason.MANDATORY_GATE_SKIPPED.value in report.reason_codes
    )


def test_unsupported_mandatory_gate_fails() -> None:
    override = GateResult(
        gate="type",
        disposition=GateDisposition.UNSUPPORTED,
        detector_id="detector:type",
        ran=True,
    )
    evidence = complete_evidence(gate_overrides={"type": override})
    report = validate_post_repair_epoch(evidence)
    assert report.ok is False
    assert (
        PostRepairValidationReason.MANDATORY_GATE_UNSUPPORTED.value
        in report.reason_codes
    )


def test_synthetic_release_child_fails() -> None:
    evidence = complete_evidence(
        proof_transition=_transition(synthetic_children=("release:synthetic-child",))
    )
    report = validate_post_repair_epoch(evidence)
    assert report.ok is False
    assert (
        PostRepairValidationReason.SYNTHETIC_RELEASE_CHILD.value in report.reason_codes
    )


def test_finding_must_close_for_intended_semantic_reason() -> None:
    evidence = complete_evidence(
        finding=_finding(semantic_reason="tests_deleted_to_silence_finding")
    )
    report = validate_post_repair_epoch(evidence)
    assert report.ok is False
    assert (
        PostRepairValidationReason.FINDING_CLOSED_WRONG_REASON.value
        in report.reason_codes
    )


def test_finding_not_closed_fails() -> None:
    evidence = complete_evidence(
        finding=_finding(closed=False, residual_finding_ids=(FINDING,))
    )
    report = validate_post_repair_epoch(evidence)
    assert report.ok is False
    assert PostRepairValidationReason.FINDING_NOT_CLOSED.value in report.reason_codes


def test_protected_invariant_regression_fails() -> None:
    evidence = complete_evidence(
        invariants=_invariants(regressed_keys=("no_llm_policy",))
    )
    report = validate_post_repair_epoch(evidence)
    assert report.ok is False
    assert (
        PostRepairValidationReason.PROTECTED_INVARIANT_REGRESSED.value
        in report.reason_codes
    )


def test_model_calls_fail_closed() -> None:
    evidence = complete_evidence(model_invocation_count=1)
    report = validate_post_repair_epoch(evidence)
    assert report.ok is False
    assert report.runtime_model_calls == 1
    assert PostRepairValidationReason.MODEL_CALLS_OBSERVED.value in report.reason_codes


def test_provider_calls_fail_closed() -> None:
    evidence = complete_evidence(provider_invocation_count=2)
    report = validate_post_repair_epoch(evidence)
    assert report.ok is False
    assert report.runtime_provider_calls == 2
    assert (
        PostRepairValidationReason.PROVIDER_CALLS_OBSERVED.value in report.reason_codes
    )


def test_failed_unit_gate_fails() -> None:
    evidence = complete_evidence(commands=_commands(unit_passed=False))
    report = validate_post_repair_epoch(evidence)
    assert report.ok is False
    assert (
        PostRepairValidationReason.MANDATORY_GATE_FAILED.value in report.reason_codes
    )


def test_incomplete_protected_invariant_set_rejected() -> None:
    with pytest.raises(PostRepairValidationError):
        ProtectedInvariantEvidence(checked_keys=("no_llm_policy",))


def test_proof_transition_requires_distinct_epochs() -> None:
    with pytest.raises(PostRepairValidationError):
        RepairProofTransition(
            pre_epoch_id="epoch:same",
            post_epoch_id="epoch:same",
            pre_proof_ids=(),
            post_proof_ids=("proof:post",),
            closed_finding_ids=(),
        )


def test_materialize_post_repair_epoch(tmp_path: Path) -> None:
    dest = tmp_path / "post-repair-epoch.json"
    payload = materialize_post_repair_epoch(destination=dest)
    assert dest.is_file()
    assert payload["runtime_model_calls"] == 0
    assert payload["runtime_provider_calls"] == 0
    assert payload["interface"] == POST_REPAIR_VALIDATION_INTERFACE
    assert payload["report"]["claims_completion"] is True
    assert payload["report"]["disposition"] == "passed"
    assert payload["grants_write_authority"] is False
    assert set(payload["mandatory_gates"]) == set(MANDATORY_GATES)


def test_report_content_identity_stable() -> None:
    report_a = validate_post_repair_epoch(complete_evidence())
    report_b = validate_post_repair_epoch(complete_evidence())
    assert report_a.content_id == report_b.content_id
    assert report_a.to_dict()["interface"] == POST_REPAIR_VALIDATION_INTERFACE
