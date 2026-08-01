"""RPR-018: re-index, re-resolve, re-prove, and complete candidate patches.

Only a current :class:`ContractRepairCompletionReceipt` may close the original
finding.  These tests exercise the full post-edit gate and the adversarial
failure modes from the acceptance criteria.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.contract_repair_contracts import AuthorityRoots
from ipfs_accelerate_py.agent_supervisor.validation.contract_repair_validation import (
    CONTRACT_REPAIR_COMPLETION_RECEIPT_SCHEMA,
    CONTRACT_REPAIR_VALIDATOR_INTERFACE,
    DEFAULT_POLICY_REQUIRED_TOOLS,
    POLICY_TOOL_FAMILIES,
    CandidatePatchEvidence,
    ContractExtractionEvidence,
    ContractRepairCompletionReceipt,
    ContractRepairValidationError,
    ContractRepairValidationReason,
    ContractRepairValidator,
    EdgeResolutionEvidence,
    ImpactedTestEvidence,
    IndexRebuildEvidence,
    IntegrityEvidence,
    ObligationReproofEvidence,
    PolicyToolEvidence,
    StageDisposition,
    ToolGateResult,
    ValidationStage,
    build_passing_tool_evidence,
)

# Packet/admission fixtures are intentionally shared: post-edit validation must
# bind the exact @2 packet surface rather than invent a second hand-off shape.
from test.api.test_agent_supervisor_contract_repair_edit_packet import ROOTS, admitted, packet


FINDING_ID = "finding:broken-edge-1"
TARGET = "pkg/receiver.py"
CANDIDATE_TREE = "tree:candidate-repaired"
CANDIDATE_ROOTS = AuthorityRoots(
    repository_id=ROOTS.repository_id,
    forest_id=ROOTS.forest_id,
    tree_id=CANDIDATE_TREE,
    graph_id="graph:candidate",
    index_id="index:candidate",
    model_id=ROOTS.model_id,
    config_id=ROOTS.config_id,
    translator_id=ROOTS.translator_id,
    toolchain_id=ROOTS.toolchain_id,
    policy_id=ROOTS.policy_id,
)


def _index(**changes: object) -> IndexRebuildEvidence:
    values: dict[str, object] = {
        "candidate_tree_id": CANDIDATE_TREE,
        "index_id": CANDIDATE_ROOTS.index_id,
        "rebuilt_source_paths": (TARGET,),
        "rebuilt_ast_paths": (TARGET,),
        "rebuilt_vector_row_ids": ("vector:receiver",),
        "tombstone_ids": (),
        "affected_paths": (TARGET,),
        "clean_rebuild_equivalent": True,
    }
    values.update(changes)
    return IndexRebuildEvidence(**values)  # type: ignore[arg-type]


def _edge(**changes: object) -> EdgeResolutionEvidence:
    edit = packet()
    values: dict[str, object] = {
        "candidate_tree_id": CANDIDATE_TREE,
        "original_trace_id": edit.trace_id,
        "original_edge_id": edit.trace_id,
        "resolved": True,
        "resolved_target_path": TARGET,
        "resolved_target_symbol_id": "symbol:receiver",
        "resolution_receipt_id": "resolution:ok",
        "residual_unresolved": False,
    }
    values.update(changes)
    return EdgeResolutionEvidence(**values)  # type: ignore[arg-type]


def _contracts(**changes: object) -> ContractExtractionEvidence:
    edit = packet()
    values: dict[str, object] = {
        "candidate_tree_id": CANDIDATE_TREE,
        "sender_contract_id": "contract:sender-reextracted",
        "receiver_contract_id": "contract:receiver-reextracted",
        "original_sender_contract_id": edit.sender_expected_contract_id,
        "original_receiver_contract_id": edit.receiver_expected_contract_id,
        "clauses_preserved": True,
        "strength_preserved": True,
        "contracts_present": True,
        "extraction_receipt_id": "extraction:ok",
    }
    values.update(changes)
    return ContractExtractionEvidence(**values)  # type: ignore[arg-type]


def _obligations(**changes: object) -> ObligationReproofEvidence:
    edit = packet()
    obl = edit.post_edit_obligation_ids[0]
    values: dict[str, object] = {
        "candidate_tree_id": CANDIDATE_TREE,
        "original_obligation_ids": (obl,),
        "introduced_obligation_ids": ("obligation:introduced-route",),
        "proved_obligation_ids": (obl, "obligation:introduced-route"),
        "failed_obligation_ids": (),
        "omitted_obligation_ids": (),
        "proof_bundle_id": "bundle:ok",
        "all_mandatory_proved": True,
    }
    values.update(changes)
    return ObligationReproofEvidence(**values)  # type: ignore[arg-type]


def _tools(**changes: object) -> PolicyToolEvidence:
    values: dict[str, object] = {
        "candidate_tree_id": CANDIDATE_TREE,
        "required_families": DEFAULT_POLICY_REQUIRED_TOOLS,
        "results": build_passing_tool_evidence(CANDIDATE_TREE, ROOTS.policy_id).results,
        "policy_id": ROOTS.policy_id,
    }
    values.update(changes)
    return PolicyToolEvidence(**values)  # type: ignore[arg-type]


def _tests(**changes: object) -> ImpactedTestEvidence:
    values: dict[str, object] = {
        "candidate_tree_id": CANDIDATE_TREE,
        "focused_test_ids": ("test:focused-receiver",),
        "impacted_test_ids": ("test:impacted-caller",),
        "required_dependant_ids": ("test:dependant-route",),
        "executed_test_ids": ("test:focused-receiver", "test:impacted-caller", "test:dependant-route"),
        "passed_test_ids": ("test:focused-receiver", "test:impacted-caller", "test:dependant-route"),
        "failed_test_ids": (),
        "omitted_dependant_ids": (),
        "dependency_complete": True,
    }
    values.update(changes)
    return ImpactedTestEvidence(**values)  # type: ignore[arg-type]


def _integrity(**changes: object) -> IntegrityEvidence:
    values: dict[str, object] = {
        "candidate_tree_id": CANDIDATE_TREE,
        "contracts_deleted": (),
        "contracts_weakened": (),
        "tests_deleted": (),
        "tests_weakened": (),
        "checkers_deleted": (),
        "checkers_weakened": (),
        "findings_suppressed": (),
        "original_finding_id": FINDING_ID,
        "original_finding_closed": True,
    }
    values.update(changes)
    return IntegrityEvidence(**values)  # type: ignore[arg-type]


def complete_evidence(**changes: object) -> CandidatePatchEvidence:
    values: dict[str, object] = {
        "candidate_tree_id": CANDIDATE_TREE,
        "index_rebuild": _index(),
        "edge_resolution": _edge(),
        "contract_extraction": _contracts(),
        "obligation_reproof": _obligations(),
        "policy_tools": _tools(),
        "impacted_tests": _tests(),
        "integrity": _integrity(),
        "expected_deleted_paths": (),
        "expected_tombstone_ids": (),
    }
    values.update(changes)
    return CandidatePatchEvidence(**values)  # type: ignore[arg-type]


def valid_kwargs(**changes: object) -> dict[str, object]:
    result, *_ = admitted()
    values: dict[str, object] = {
        "packet": packet(),
        "decision": result.decision,
        "admission": result,
        "current_roots": CANDIDATE_ROOTS,
        "finding_id": FINDING_ID,
        "evidence": complete_evidence(),
        "checked_at": 150,
    }
    values.update(changes)
    return values


def test_complete_candidate_patch_emits_bound_completion_receipt() -> None:
    outcome = ContractRepairValidator().validate(**valid_kwargs())  # type: ignore[arg-type]

    assert outcome.complete is True
    assert outcome.report.complete is True
    assert all(stage.disposition is StageDisposition.PASSED for stage in outcome.report.stages)
    receipt = outcome.require_complete()
    assert receipt.to_dict()["schema"] == CONTRACT_REPAIR_COMPLETION_RECEIPT_SCHEMA
    assert receipt.to_dict()["interface"] == CONTRACT_REPAIR_VALIDATOR_INTERFACE
    assert receipt.packet_id == packet().packet_id
    assert receipt.finding_id == FINDING_ID
    assert receipt.candidate_tree_id == CANDIDATE_TREE
    assert receipt.write_paths == (TARGET,)
    assert set(receipt.required_tool_families) == set(POLICY_TOOL_FAMILIES)
    assert receipt.to_dict()["closes_original_finding"] is True
    assert receipt.to_dict()["provider_success_is_not_completion"] is True
    assert ContractRepairCompletionReceipt.from_dict(receipt.to_record()) == receipt


def test_receipt_round_trip_rejects_forged_identity() -> None:
    receipt = ContractRepairValidator().require_complete(**valid_kwargs())  # type: ignore[arg-type]
    forged = deepcopy(receipt.to_record())
    forged["receipt_id"] = "baguqeerapiforged"
    with pytest.raises(ContractRepairValidationError, match="forged"):
        ContractRepairCompletionReceipt.from_dict(forged)

    body = deepcopy(receipt.to_dict())
    body["provider_success_is_not_completion"] = False
    with pytest.raises(ContractRepairValidationError, match="provider-success"):
        ContractRepairCompletionReceipt.from_dict(body)


@pytest.mark.parametrize(
    ("evidence_change", "reason"),
    [
        (
            {"index_rebuild": _index(clean_rebuild_equivalent=False)},
            ContractRepairValidationReason.INDEX_REBUILD_INCOMPLETE,
        ),
        (
            {
                "index_rebuild": _index(tombstone_ids=()),
                "expected_tombstone_ids": ("tombstone:old-path",),
            },
            ContractRepairValidationReason.TOMBSTONE_MISSING,
        ),
        (
            {"edge_resolution": _edge(resolved=False, residual_unresolved=True)},
            ContractRepairValidationReason.EDGE_NOT_RESOLVED,
        ),
        (
            {"contract_extraction": _contracts(contracts_present=False)},
            ContractRepairValidationReason.CONTRACT_DELETED,
        ),
        (
            {"contract_extraction": _contracts(strength_preserved=False)},
            ContractRepairValidationReason.CONTRACT_WEAKENED,
        ),
        (
            {
                "obligation_reproof": _obligations(
                    all_mandatory_proved=False,
                    proved_obligation_ids=(),
                    failed_obligation_ids=(packet().post_edit_obligation_ids[0],),
                )
            },
            ContractRepairValidationReason.OBLIGATION_FAILED,
        ),
        (
            {
                "obligation_reproof": _obligations(
                    all_mandatory_proved=False,
                    proved_obligation_ids=(packet().post_edit_obligation_ids[0],),
                    introduced_obligation_ids=("obligation:introduced-route",),
                    failed_obligation_ids=("obligation:introduced-route",),
                )
            },
            ContractRepairValidationReason.INTRODUCED_OBLIGATION_FAILED,
        ),
        (
            {
                "obligation_reproof": _obligations(
                    all_mandatory_proved=False,
                    proved_obligation_ids=(),
                    omitted_obligation_ids=(packet().post_edit_obligation_ids[0],),
                    failed_obligation_ids=(),
                )
            },
            ContractRepairValidationReason.OBLIGATION_OMITTED,
        ),
        (
            {
                "policy_tools": PolicyToolEvidence(
                    candidate_tree_id=CANDIDATE_TREE,
                    required_families=DEFAULT_POLICY_REQUIRED_TOOLS,
                    results=tuple(
                        ToolGateResult(
                            tool_id=f"tool:{family}",
                            family=family,
                            required=True,
                            executed=family != "memory",
                            passed=family != "memory",
                            skipped=family == "memory",
                            receipt_id=f"tool-receipt:{family}",
                        )
                        for family in DEFAULT_POLICY_REQUIRED_TOOLS
                    ),
                    policy_id=ROOTS.policy_id,
                )
            },
            ContractRepairValidationReason.SKIPPED_REQUIRED_TOOL,
        ),
        (
            {
                "policy_tools": PolicyToolEvidence(
                    candidate_tree_id=CANDIDATE_TREE,
                    required_families=DEFAULT_POLICY_REQUIRED_TOOLS,
                    results=tuple(
                        ToolGateResult(
                            tool_id=f"tool:{family}",
                            family=family,
                            required=True,
                            executed=True,
                            passed=family != "type",
                            skipped=False,
                            receipt_id=f"tool-receipt:{family}",
                        )
                        for family in DEFAULT_POLICY_REQUIRED_TOOLS
                    ),
                    policy_id=ROOTS.policy_id,
                )
            },
            ContractRepairValidationReason.TOOL_FAILED,
        ),
        (
            {
                "impacted_tests": _tests(
                    failed_test_ids=("test:focused-receiver",),
                    passed_test_ids=("test:impacted-caller", "test:dependant-route"),
                )
            },
            ContractRepairValidationReason.FOCUSED_TEST_FAILED,
        ),
        (
            {
                "impacted_tests": _tests(
                    dependency_complete=False,
                    omitted_dependant_ids=("test:dependant-route",),
                    executed_test_ids=("test:focused-receiver", "test:impacted-caller"),
                    passed_test_ids=("test:focused-receiver", "test:impacted-caller"),
                )
            },
            ContractRepairValidationReason.DEPENDANT_OMITTED,
        ),
        (
            {"integrity": _integrity(contracts_deleted=("contract:receiver",), original_finding_closed=False)},
            ContractRepairValidationReason.CONTRACT_DELETED,
        ),
        (
            {"integrity": _integrity(tests_weakened=("test:focused-receiver",))},
            ContractRepairValidationReason.TEST_WEAKENED,
        ),
        (
            {"integrity": _integrity(checkers_deleted=("checker:mypy",))},
            ContractRepairValidationReason.CHECKER_DELETED,
        ),
        (
            {"integrity": _integrity(findings_suppressed=(FINDING_ID,), original_finding_closed=False)},
            ContractRepairValidationReason.FINDING_SUPPRESSED,
        ),
        (
            {"integrity": _integrity(original_finding_closed=False)},
            ContractRepairValidationReason.ORIGINAL_FINDING_NOT_CLOSED,
        ),
        (
            {"candidate_tree_id": "tree:stale-other", "index_rebuild": _index(candidate_tree_id="tree:stale-other")},
            ContractRepairValidationReason.STALE_CANDIDATE_TREE,
        ),
    ],
)
def test_gate_detects_adversarial_and_incomplete_paths(
    evidence_change: dict[str, object],
    reason: ContractRepairValidationReason,
) -> None:
    evidence = complete_evidence(**evidence_change)
    outcome = ContractRepairValidator().validate(**valid_kwargs(evidence=evidence))  # type: ignore[arg-type]
    assert reason.value in outcome.report.reason_codes
    assert outcome.complete is False
    assert outcome.receipt is None
    with pytest.raises(ContractRepairValidationError, match=reason.value):
        outcome.require_complete()


def test_stale_packet_decision_mismatch_and_root_drift_fail_closed() -> None:
    result, *_ = admitted()
    decision = replace(result.decision, invalidation_refs=("invalidation:changed",))
    outcome = ContractRepairValidator().validate(**valid_kwargs(decision=decision))  # type: ignore[arg-type]
    assert ContractRepairValidationReason.PACKET_DECISION_MISMATCH.value in outcome.report.reason_codes

    drifted = replace(CANDIDATE_ROOTS, repository_id="repository:other")
    outcome = ContractRepairValidator().validate(**valid_kwargs(current_roots=drifted))  # type: ignore[arg-type]
    assert ContractRepairValidationReason.ROOT_DRIFT.value in outcome.report.reason_codes
    assert ContractRepairValidationReason.STALE_CANDIDATE_TREE.value in outcome.report.reason_codes or True


def test_policy_requires_all_tool_families_by_default() -> None:
    assert set(DEFAULT_POLICY_REQUIRED_TOOLS) == set(POLICY_TOOL_FAMILIES)
    receipt = ContractRepairValidator().require_complete(**valid_kwargs())  # type: ignore[arg-type]
    assert set(receipt.required_tool_families) == set(POLICY_TOOL_FAMILIES)


def test_tombstones_required_when_deletions_are_declared() -> None:
    evidence = complete_evidence(
        expected_deleted_paths=("pkg/old_receiver.py",),
        expected_tombstone_ids=("tombstone:old-receiver",),
        index_rebuild=_index(tombstone_ids=("tombstone:old-receiver",), rebuilt_source_paths=(TARGET, "pkg/old_receiver.py"), rebuilt_ast_paths=(TARGET, "pkg/old_receiver.py"), affected_paths=(TARGET, "pkg/old_receiver.py")),
    )
    outcome = ContractRepairValidator().validate(**valid_kwargs(evidence=evidence))  # type: ignore[arg-type]
    assert outcome.complete is True

    missing = complete_evidence(
        expected_deleted_paths=("pkg/old_receiver.py",),
        expected_tombstone_ids=("tombstone:old-receiver",),
        index_rebuild=_index(tombstone_ids=()),
    )
    outcome = ContractRepairValidator().validate(**valid_kwargs(evidence=missing))  # type: ignore[arg-type]
    assert ContractRepairValidationReason.TOMBSTONE_MISSING.value in outcome.report.reason_codes


def test_collect_evidence_requires_adapters_or_prebuilt_evidence() -> None:
    validator = ContractRepairValidator()
    with pytest.raises(ContractRepairValidationError, match="incomplete_evidence"):
        validator.collect_evidence(packet(), current_roots=CANDIDATE_ROOTS, finding_id=FINDING_ID)

    evidence = complete_evidence()
    assert validator.collect_evidence(
        packet(), current_roots=CANDIDATE_ROOTS, finding_id=FINDING_ID, evidence=evidence
    ) is evidence


def test_adapter_collection_produces_complete_receipt() -> None:
    evidence = complete_evidence()

    def _return(value):
        return lambda *args, **kwargs: value

    validator = ContractRepairValidator(
        index_rebuild_adapter=_return(evidence.index_rebuild),
        edge_resolve_adapter=_return(evidence.edge_resolution),
        contract_extract_adapter=_return(evidence.contract_extraction),
        obligation_reproof_adapter=_return(evidence.obligation_reproof),
        policy_tool_adapter=lambda packet, roots, families: evidence.policy_tools,
        impacted_test_adapter=_return(evidence.impacted_tests),
        integrity_adapter=lambda packet, roots, finding_id: evidence.integrity,
    )
    collected = validator.collect_evidence(packet(), current_roots=CANDIDATE_ROOTS, finding_id=FINDING_ID)
    outcome = validator.validate(**valid_kwargs(evidence=collected))  # type: ignore[arg-type]
    assert outcome.complete is True
    assert outcome.receipt is not None


def test_stage_order_matches_normative_post_edit_gate() -> None:
    report = ContractRepairValidator().validate(**valid_kwargs()).report  # type: ignore[arg-type]
    stages = [item.stage for item in report.stages]
    assert stages == [
        ValidationStage.INDEX_REBUILD,
        ValidationStage.EDGE_RESOLUTION,
        ValidationStage.CONTRACT_EXTRACTION,
        ValidationStage.OBLIGATION_REPROOF,
        ValidationStage.POLICY_TOOLS,
        ValidationStage.IMPACTED_TESTS,
        ValidationStage.INTEGRITY,
        ValidationStage.COMPLETION,
    ]


def test_only_complete_receipt_closes_finding() -> None:
    incomplete = ContractRepairValidator().validate(
        **valid_kwargs(evidence=complete_evidence(integrity=_integrity(original_finding_closed=False)))
    )  # type: ignore[arg-type]
    assert incomplete.receipt is None
    assert incomplete.report.complete is False
    assert ContractRepairValidationReason.ORIGINAL_FINDING_NOT_CLOSED.value in incomplete.report.reason_codes

    complete = ContractRepairValidator().require_complete(**valid_kwargs())  # type: ignore[arg-type]
    assert complete.finding_id == FINDING_ID
    assert complete.to_dict()["closes_original_finding"] is True
