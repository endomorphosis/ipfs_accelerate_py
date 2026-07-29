"""Contract tests for the final, non-dispatching repair provider gate."""

from __future__ import annotations

from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.repository_snapshot import (
    CoverageDisposition,
    CoverageKind,
    EntryKind,
    GitStatus,
    RepositorySnapshot,
    RepositorySnapshotStats,
)
from ipfs_accelerate_py.agent_supervisor.analysis.contract_repair_reranker import CandidateEligibilityDisposition
from ipfs_accelerate_py.agent_supervisor.integrations.contract_repair_capabilities import (
    ContractRepairCapability,
    ContractRepairCapabilityDiagnostic,
    ContractRepairCapabilityReport,
    ContractRepairCapabilityStatus,
    ContractRepairDiagnosticCode,
)
from ipfs_accelerate_py.agent_supervisor.validation.contract_repair_pre_provider_gate import (
    ContractRepairPreProviderGate,
    ContractRepairPreProviderGateError,
    PreProviderGateReason,
    PreProviderGateReceipt,
)

# The packet fixture constructs the complete admitted target boundary and is
# intentionally reused here: this gate must not invent a second packet shape.
from test.api.test_agent_supervisor_contract_repair_edit_packet import ROOTS, admitted, packet


TARGET = "pkg/receiver.py"
ARTIFACT = "blob:receiver"


def snapshot(**changes: object) -> RepositorySnapshot:
    disposition = CoverageDisposition(
        TARGET, CoverageKind.SEMANTIC_AST, GitStatus.CLEAN, EntryKind.REGULAR,
        "semantic_source", "fixture", content_digest="sha256:receiver", git_object_id=ARTIFACT,
    )
    values: dict[str, object] = {
        "primary_root": ".", "head_commit_id": "commit:test", "head_tree_id": ROOTS.tree_id,
        "index_tree_id": ROOTS.tree_id, "scope_policy_id": "scope:test", "scope_id": "scope:test",
        "dispositions": (disposition,), "dependency_identities": (), "gitlinks": (),
        "stats": RepositorySnapshotStats(1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1),
    }
    values.update(changes)
    return RepositorySnapshot(**values)  # type: ignore[arg-type]


def capabilities(*, complete: bool = True) -> ContractRepairCapabilityReport:
    if complete:
        cap = ContractRepairCapability(
            "datasets.hammer", ContractRepairCapabilityStatus.AVAILABLE,
            module_paths=("fixture.hammer",), reconstruction_compatible=True,
        )
    else:
        cap = ContractRepairCapability(
            "datasets.hammer", ContractRepairCapabilityStatus.PARTIAL,
            diagnostic=ContractRepairCapabilityDiagnostic(
                ContractRepairDiagnosticCode.PARTIAL_INTERFACE, "datasets.hammer", "incomplete",
            ),
        )
    return ContractRepairCapabilityReport((cap,), (), (), "gitlink:test")


def valid_kwargs(**changes: object) -> dict[str, object]:
    result, *_ = admitted()
    values: dict[str, object] = {
        "packet": packet(), "decision": result.decision, "admission": result,
        "snapshot": snapshot(), "current_roots": ROOTS, "capability_report": capabilities(), "now": 150,
    }
    values.update(changes)
    return values


def test_current_admitted_packet_emits_bounded_non_dispatch_receipt() -> None:
    receipt = ContractRepairPreProviderGate().require_valid(**valid_kwargs())  # type: ignore[arg-type]

    assert receipt.target_path == TARGET
    assert receipt.write_paths == (TARGET,)
    assert receipt.to_dict()["provider_invoked"] is False
    assert receipt.to_dict()["authorized_paths"] == [TARGET]
    assert PreProviderGateReceipt.from_dict(receipt.to_record()) == receipt


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"snapshot": snapshot(index_tree_id="tree:changed")}, PreProviderGateReason.TREE_OR_OVERLAY_CHANGED),
        ({"snapshot": snapshot(dispositions=())}, PreProviderGateReason.TARGET_MISSING_OR_MOVED),
        ({"snapshot": snapshot(dispositions=(CoverageDisposition(TARGET, CoverageKind.SEMANTIC_AST, GitStatus.CLEAN, EntryKind.REGULAR, "semantic", "fixture", git_object_id="blob:other"),))}, PreProviderGateReason.TARGET_HASH_DRIFT),
        ({"current_roots": replace(ROOTS, policy_id="policy:changed")}, PreProviderGateReason.ROOT_DRIFT),
        ({"now": 200}, PreProviderGateReason.EXPIRED_PROOF),
        ({"capability_report": capabilities(complete=False)}, PreProviderGateReason.INCOMPLETE_CAPABILITY),
        ({"read_only_paths": (TARGET,)}, PreProviderGateReason.READ_ONLY_OR_ESCAPED_PATH),
    ],
)
def test_gate_rejects_drift_before_a_provider_can_be_called(change: dict[str, object], reason: PreProviderGateReason) -> None:
    reasons = ContractRepairPreProviderGate().validate(**valid_kwargs(**change))  # type: ignore[arg-type]
    assert reason in reasons
    with pytest.raises(ContractRepairPreProviderGateError, match=reason.value):
        ContractRepairPreProviderGate().require_valid(**valid_kwargs(**change))  # type: ignore[arg-type]


def test_packet_decision_mismatch_and_abstention_fail_closed() -> None:
    values = valid_kwargs()
    decision = values["decision"]
    assert decision is not None
    values["decision"] = replace(decision, invalidation_refs=("invalidation:changed",))  # type: ignore[arg-type]
    assert PreProviderGateReason.PACKET_DECISION_MISMATCH in ContractRepairPreProviderGate().validate(**values)  # type: ignore[arg-type]

    result, *_ = admitted()
    rejected = replace(result.decision, disposition="rejected", strategy="reject", selected_candidate_id="", permitted_write_paths=())
    assert PreProviderGateReason.AMBIGUOUS_OR_ABSTAINED in ContractRepairPreProviderGate().validate(
        **valid_kwargs(decision=rejected)  # type: ignore[arg-type]
    )


def test_downgraded_rerank_proof_is_rejected_even_with_the_original_packet() -> None:
    result, *_ = admitted()
    rank = result.audit.ranks[0]
    downgraded_audit = replace(
        result.audit,
        ranks=(replace(rank, disposition=CandidateEligibilityDisposition.INELIGIBLE, reason_codes=("proof_missing",)),),
    )
    reasons = ContractRepairPreProviderGate().validate(
        **valid_kwargs(admission=replace(result, audit=downgraded_audit))  # type: ignore[arg-type]
    )
    assert PreProviderGateReason.PROOF_DOWNGRADED in reasons


def test_receipt_rejects_forged_identity_and_path_broadening() -> None:
    receipt = ContractRepairPreProviderGate().require_valid(**valid_kwargs())  # type: ignore[arg-type]
    forged = receipt.to_record()
    forged["authorized_paths"] = [TARGET, "pkg/other.py"]
    with pytest.raises(ContractRepairPreProviderGateError, match="broaden"):
        PreProviderGateReceipt.from_dict(forged)
