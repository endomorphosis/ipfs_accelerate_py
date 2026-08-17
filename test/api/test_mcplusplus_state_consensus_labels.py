"""MCPP-039: ConsensusPlugin@1 honest guarantee labels.

Acceptance:
  - Tests fail if a neighborhood result is labeled BFT.
  - Deterministic test adapter is supplied.
  - Labels distinguish coordination, majority approval, crash consensus, and BFT.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from ipfs_accelerate_py.mcp_server.mcplusplus.state.consensus_plugin import (
    CONSENSUS_EVIDENCE_SCHEMA,
    CONSENSUS_PLUGIN_INTERFACE,
    CONSENSUS_RESULT_SCHEMA,
    DETERMINISTIC_TEST_ADAPTER_PLUGIN_ID,
    GUARANTEE_BFT,
    GUARANTEE_COORDINATION,
    GUARANTEE_CRASH_CONSENSUS,
    GUARANTEE_LABELS,
    GUARANTEE_MAJORITY_APPROVAL,
    PROFILE_G_ALLOWED_GUARANTEES,
    PROFILE_G_NEIGHBORHOOD_PLUGIN_ID,
    VERDICT_ABSTAIN,
    VERDICT_CHALLENGE,
    VERDICT_SUPPORT,
    ConsensusEvidenceError,
    ConsensusPluginError,
    ConsensusRejectedError,
    DeterministicTestAdapter,
    GuaranteeLabel,
    InvalidGuaranteeError,
    NeighborhoodGuaranteeError,
    build_plugin_evidence,
    evaluate_majority,
    is_profile_g_allowed_guarantee,
    label_neighborhood_result,
    normalize_guarantee_label,
    require_plugin_guarantee,
    require_profile_g_guarantee,
    validate_plugin_evidence,
    wire_neighborhood_result,
)

PROPOSAL_CID = "bafkreifzjut3te2nhyekklss27nh3k72ysco7y32koao5eei66wof36n5e"
STATE_ID = "state:test/consensus-labels"
PEERS = ("did:key:peer-a", "did:key:peer-b", "did:key:peer-c")

SPEC_PATH = (
    Path(__file__).resolve().parents[2]
    / "ipfs_accelerate_py"
    / "mcplusplus"
    / "docs"
    / "spec"
    / "consensus-plugin.md"
)


# ---------------------------------------------------------------------------
# Closed label set
# ---------------------------------------------------------------------------


def test_guarantee_labels_are_exactly_four_honest_classes() -> None:
    assert GUARANTEE_LABELS == frozenset(
        {
            GUARANTEE_COORDINATION,
            GUARANTEE_MAJORITY_APPROVAL,
            GUARANTEE_CRASH_CONSENSUS,
            GUARANTEE_BFT,
        }
    )
    assert {label.value for label in GuaranteeLabel} == set(GUARANTEE_LABELS)


@pytest.mark.parametrize(
    "label",
    sorted(GUARANTEE_LABELS),
)
def test_normalize_accepts_each_honest_label(label: str) -> None:
    assert normalize_guarantee_label(label) == label
    assert normalize_guarantee_label(GuaranteeLabel(label)) == label


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "   ",
        "BFT",
        "byzantine",
        "raft",
        "quorum",
        "majority",
        "consensus",
        None,
        1,
        ["bft"],
    ],
)
def test_unknown_or_alias_guarantee_labels_are_rejected(bad: object) -> None:
    with pytest.raises(InvalidGuaranteeError):
        normalize_guarantee_label(bad)


# ---------------------------------------------------------------------------
# Profile G neighborhood must not be labeled BFT (acceptance)
# ---------------------------------------------------------------------------


def test_neighborhood_result_labeled_bft_fails_closed() -> None:
    """Acceptance: tests fail if a neighborhood result is labeled BFT."""

    with pytest.raises(NeighborhoodGuaranteeError) as excinfo:
        label_neighborhood_result(GUARANTEE_BFT)
    message = str(excinfo.value).lower()
    assert "bft" in message
    assert "neighborhood" in message


def test_wire_neighborhood_rejects_bft_guarantee() -> None:
    attestations = [
        {"attester_did": peer, "verdict": VERDICT_SUPPORT} for peer in PEERS
    ]
    with pytest.raises(NeighborhoodGuaranteeError):
        wire_neighborhood_result(
            state_id=STATE_ID,
            proposal_cid=PROPOSAL_CID,
            attestations=attestations,
            members=PEERS,
            guarantee=GUARANTEE_BFT,
        )


def test_wire_neighborhood_rejects_crash_consensus_guarantee() -> None:
    """Neighborhood is coordination / majority only — not CFT either."""

    attestations = [
        {"attester_did": peer, "verdict": VERDICT_SUPPORT} for peer in PEERS
    ]
    with pytest.raises(NeighborhoodGuaranteeError):
        wire_neighborhood_result(
            state_id=STATE_ID,
            proposal_cid=PROPOSAL_CID,
            attestations=attestations,
            members=PEERS,
            guarantee=GUARANTEE_CRASH_CONSENSUS,
        )


def test_build_neighborhood_evidence_rejects_bft() -> None:
    with pytest.raises(NeighborhoodGuaranteeError):
        build_plugin_evidence(
            plugin_id=PROFILE_G_NEIGHBORHOOD_PLUGIN_ID,
            guarantee=GUARANTEE_BFT,
            state_id=STATE_ID,
            proposal_cid=PROPOSAL_CID,
            evidence_kind="neighborhood",
            members=PEERS,
            approvals=PEERS,
            profile_g_neighborhood=True,
        )


def test_validate_plugin_evidence_rejects_neighborhood_bft_mapping() -> None:
    payload = {
        "schema": CONSENSUS_EVIDENCE_SCHEMA,
        "plugin_id": PROFILE_G_NEIGHBORHOOD_PLUGIN_ID,
        "guarantee": GUARANTEE_BFT,
        "state_id": STATE_ID,
        "proposal_cid": PROPOSAL_CID,
        "evidence_kind": "neighborhood",
        "members": list(PEERS),
        "approvals": list(PEERS),
        "rejections": [],
        "abstentions": [],
        "threshold": 2,
        "round_id": "1",
        "source": "profile_g_neighborhood",
    }
    with pytest.raises(NeighborhoodGuaranteeError):
        validate_plugin_evidence(payload)


def test_require_profile_g_guarantee_allows_only_coordination_and_majority() -> None:
    assert require_profile_g_guarantee(GUARANTEE_COORDINATION) == GUARANTEE_COORDINATION
    assert (
        require_profile_g_guarantee(GUARANTEE_MAJORITY_APPROVAL)
        == GUARANTEE_MAJORITY_APPROVAL
    )
    assert PROFILE_G_ALLOWED_GUARANTEES == frozenset(
        {GUARANTEE_COORDINATION, GUARANTEE_MAJORITY_APPROVAL}
    )
    assert is_profile_g_allowed_guarantee(GUARANTEE_BFT) is False
    assert is_profile_g_allowed_guarantee(GUARANTEE_CRASH_CONSENSUS) is False


# ---------------------------------------------------------------------------
# Valid neighborhood wiring under honest labels
# ---------------------------------------------------------------------------


def test_neighborhood_majority_approval_accepts_simple_majority() -> None:
    attestations = [
        {"attester_did": "did:key:peer-a", "verdict": VERDICT_SUPPORT},
        {"attester_did": "did:key:peer-b", "verdict": VERDICT_SUPPORT},
        {"attester_did": "did:key:peer-c", "verdict": VERDICT_ABSTAIN},
    ]
    result = wire_neighborhood_result(
        state_id=STATE_ID,
        proposal_cid=PROPOSAL_CID,
        attestations=attestations,
        members=PEERS,
        guarantee=GUARANTEE_MAJORITY_APPROVAL,
        round_id="epoch-1",
    )
    assert result.accepted is True
    assert result.guarantee == GUARANTEE_MAJORITY_APPROVAL
    assert result.evidence_kind == "neighborhood"
    assert result.plugin_id == PROFILE_G_NEIGHBORHOOD_PLUGIN_ID
    assert result.approval_count == 2
    assert result.threshold == 2
    assert result.schema == CONSENSUS_RESULT_SCHEMA
    assert result.evidence is not None
    assert result.evidence["guarantee"] == GUARANTEE_MAJORITY_APPROVAL
    assert result.evidence["source"] == "profile_g_neighborhood"
    assert "bft" not in result.to_dict()["guarantee"]


def test_neighborhood_majority_approval_rejects_insufficient_approvals() -> None:
    attestations = [
        {"attester_did": "did:key:peer-a", "verdict": VERDICT_SUPPORT},
        {"attester_did": "did:key:peer-b", "verdict": VERDICT_CHALLENGE},
        {"attester_did": "did:key:peer-c", "verdict": VERDICT_ABSTAIN},
    ]
    result = wire_neighborhood_result(
        state_id=STATE_ID,
        proposal_cid=PROPOSAL_CID,
        attestations=attestations,
        members=PEERS,
        guarantee=GUARANTEE_MAJORITY_APPROVAL,
    )
    assert result.accepted is False
    assert result.guarantee == GUARANTEE_MAJORITY_APPROVAL
    assert result.approval_count == 1
    assert result.threshold == 2


def test_neighborhood_coordination_is_best_effort_not_bft() -> None:
    attestations = [
        {"attester_did": "did:key:peer-a", "verdict": VERDICT_SUPPORT},
    ]
    result = wire_neighborhood_result(
        state_id=STATE_ID,
        proposal_cid=PROPOSAL_CID,
        attestations=attestations,
        members=PEERS,
        guarantee=GUARANTEE_COORDINATION,
    )
    assert result.accepted is True
    assert result.guarantee == GUARANTEE_COORDINATION
    assert result.threshold == 0
    assert result.evidence_kind == "neighborhood"


# ---------------------------------------------------------------------------
# Deterministic test adapter
# ---------------------------------------------------------------------------


def test_deterministic_test_adapter_is_supplied() -> None:
    adapter = DeterministicTestAdapter(guarantee=GUARANTEE_MAJORITY_APPROVAL)
    assert adapter.interface == CONSENSUS_PLUGIN_INTERFACE
    assert adapter.plugin_id == DETERMINISTIC_TEST_ADAPTER_PLUGIN_ID
    assert adapter.guarantee == GUARANTEE_MAJORITY_APPROVAL
    assert adapter.implements_bft is False
    assert adapter.mode == "consensus"
    descriptor = adapter.describe()
    assert descriptor["implements_bft"] is False
    assert "bft" not in descriptor["guarantee"]


def test_deterministic_adapter_rejects_bft_construction() -> None:
    with pytest.raises(InvalidGuaranteeError):
        DeterministicTestAdapter(guarantee=GUARANTEE_BFT)
    with pytest.raises(InvalidGuaranteeError):
        DeterministicTestAdapter(guarantee=GUARANTEE_CRASH_CONSENSUS)


def test_deterministic_adapter_majority_is_order_independent() -> None:
    members = ["n3", "n1", "n2", "n4", "n5"]
    ballots_a = [
        ("n1", VERDICT_SUPPORT),
        ("n5", VERDICT_SUPPORT),
        ("n2", VERDICT_CHALLENGE),
        ("n3", VERDICT_SUPPORT),
        ("n4", VERDICT_ABSTAIN),
    ]
    ballots_b = list(reversed(ballots_a))

    def run(ballots: list[tuple[str, str]]):
        adapter = DeterministicTestAdapter(guarantee=GUARANTEE_MAJORITY_APPROVAL)
        evidence = adapter.propose(
            state_id=STATE_ID,
            proposal_cid=PROPOSAL_CID,
            members=members,
            round_id="r1",
        )
        for principal, verdict in ballots:
            evidence = adapter.record_ballot(
                evidence, principal=principal, verdict=verdict
            )
        return adapter.evaluate(evidence)

    result_a = run(ballots_a)
    result_b = run(ballots_b)
    assert result_a.accepted is True
    assert result_b.accepted is True
    assert result_a.to_dict() == result_b.to_dict()
    # 3 supports among 5 members → threshold 3.
    assert result_a.approval_count == 3
    assert result_a.threshold == 3


def test_deterministic_adapter_accept_raises_when_rejected() -> None:
    adapter = DeterministicTestAdapter(guarantee=GUARANTEE_MAJORITY_APPROVAL)
    evidence = adapter.propose(
        state_id=STATE_ID,
        proposal_cid=PROPOSAL_CID,
        members=PEERS,
    )
    evidence = adapter.record_ballot(
        evidence, principal="did:key:peer-a", verdict=VERDICT_SUPPORT
    )
    with pytest.raises(ConsensusRejectedError):
        adapter.accept(evidence)


def test_deterministic_adapter_neighborhood_path_fails_on_bft_label() -> None:
    adapter = DeterministicTestAdapter(guarantee=GUARANTEE_MAJORITY_APPROVAL)
    attestations = [
        {"attester_did": peer, "verdict": VERDICT_SUPPORT} for peer in PEERS
    ]
    with pytest.raises(NeighborhoodGuaranteeError):
        adapter.evaluate_neighborhood(
            state_id=STATE_ID,
            proposal_cid=PROPOSAL_CID,
            attestations=attestations,
            members=PEERS,
            guarantee=GUARANTEE_BFT,
        )


def test_deterministic_adapter_neighborhood_majority_happy_path() -> None:
    adapter = DeterministicTestAdapter(guarantee=GUARANTEE_MAJORITY_APPROVAL)
    attestations = [
        {"attester_did": "did:key:peer-a", "verdict": VERDICT_SUPPORT},
        {"attester_did": "did:key:peer-b", "verdict": VERDICT_SUPPORT},
        {"attester_did": "did:key:peer-c", "verdict": VERDICT_CHALLENGE},
    ]
    result = adapter.evaluate_neighborhood(
        state_id=STATE_ID,
        proposal_cid=PROPOSAL_CID,
        attestations=attestations,
        members=PEERS,
    )
    assert result.accepted is True
    assert result.guarantee == GUARANTEE_MAJORITY_APPROVAL
    assert result.evidence_kind == "neighborhood"


def test_deterministic_adapter_coordination_mode() -> None:
    adapter = DeterministicTestAdapter(guarantee=GUARANTEE_COORDINATION)
    evidence = adapter.propose(
        state_id=STATE_ID,
        proposal_cid=PROPOSAL_CID,
        members=PEERS,
    )
    evidence = adapter.record_ballot(
        evidence, principal="did:key:peer-b", verdict=VERDICT_SUPPORT
    )
    result = adapter.accept(evidence)
    assert result.accepted is True
    assert result.guarantee == GUARANTEE_COORDINATION
    assert result.threshold == 0


# ---------------------------------------------------------------------------
# Evidence format and plugin-level BFT honesty
# ---------------------------------------------------------------------------


def test_build_and_validate_plugin_evidence_round_trip() -> None:
    evidence = build_plugin_evidence(
        plugin_id="mcp++/consensus/raft-local@1",
        guarantee=GUARANTEE_CRASH_CONSENSUS,
        state_id=STATE_ID,
        proposal_cid=PROPOSAL_CID,
        evidence_kind="plugin",
        members=("n1", "n2", "n3"),
        approvals=("n1", "n2"),
        round_id="term-1",
        source="raft_local",
    )
    assert evidence.schema == CONSENSUS_EVIDENCE_SCHEMA
    assert evidence.guarantee == GUARANTEE_CRASH_CONSENSUS
    assert evidence.threshold == 2
    assert evidence.members == ("n1", "n2", "n3")
    restored = validate_plugin_evidence(evidence.to_dict())
    assert restored.to_dict() == evidence.to_dict()


def test_bft_requires_implements_bft_flag() -> None:
    with pytest.raises(InvalidGuaranteeError):
        require_plugin_guarantee(GUARANTEE_BFT, implements_bft=False)
    assert (
        require_plugin_guarantee(GUARANTEE_BFT, implements_bft=True) == GUARANTEE_BFT
    )
    with pytest.raises(InvalidGuaranteeError):
        build_plugin_evidence(
            plugin_id="mcp++/consensus/fake-bft@1",
            guarantee=GUARANTEE_BFT,
            state_id=STATE_ID,
            proposal_cid=PROPOSAL_CID,
            members=PEERS,
            approvals=PEERS,
            implements_bft=False,
        )
    # Explicit BFT engine path is allowed only when implements_bft=True.
    evidence = build_plugin_evidence(
        plugin_id="mcp++/consensus/real-bft@1",
        guarantee=GUARANTEE_BFT,
        state_id=STATE_ID,
        proposal_cid=PROPOSAL_CID,
        members=PEERS,
        approvals=PEERS,
        implements_bft=True,
    )
    assert evidence.guarantee == GUARANTEE_BFT
    assert evidence.evidence_kind == "plugin"


def test_metadata_cannot_redeclare_conflicting_guarantee() -> None:
    with pytest.raises(ConsensusEvidenceError):
        build_plugin_evidence(
            plugin_id=DETERMINISTIC_TEST_ADAPTER_PLUGIN_ID,
            guarantee=GUARANTEE_MAJORITY_APPROVAL,
            state_id=STATE_ID,
            proposal_cid=PROPOSAL_CID,
            members=PEERS,
            metadata={"guarantee": GUARANTEE_BFT},
        )


def test_evaluate_majority_ignores_non_member_approvals() -> None:
    accepted, count, thr = evaluate_majority(
        members=PEERS,
        approvals=("did:key:peer-a", "did:key:outsider", "did:key:peer-b"),
    )
    assert accepted is True
    assert count == 2
    assert thr == 2


def test_non_member_ballot_rejected_by_adapter() -> None:
    adapter = DeterministicTestAdapter()
    evidence = adapter.propose(
        state_id=STATE_ID,
        proposal_cid=PROPOSAL_CID,
        members=PEERS,
    )
    with pytest.raises(ConsensusEvidenceError):
        adapter.record_ballot(
            evidence, principal="did:key:outsider", verdict=VERDICT_SUPPORT
        )


# ---------------------------------------------------------------------------
# Spec document presence
# ---------------------------------------------------------------------------


def test_consensus_plugin_spec_document_exists_and_states_non_bft_neighborhood() -> None:
    assert SPEC_PATH.is_file(), f"missing spec: {SPEC_PATH}"
    text = SPEC_PATH.read_text(encoding="utf-8")
    assert "ConsensusPlugin@1" in text
    assert "coordination" in text
    assert "majority_approval" in text
    assert "crash_consensus" in text
    assert "bft" in text.lower()
    assert "Profile G" in text or "profile G" in text
    # Honest non-claim must be explicit in normative prose.
    assert "not" in text.lower() and "bft" in text.lower()
    assert "Deterministic" in text or "deterministic" in text


def test_error_hierarchy_is_usable_for_callers() -> None:
    assert issubclass(NeighborhoodGuaranteeError, InvalidGuaranteeError)
    assert issubclass(InvalidGuaranteeError, ConsensusPluginError)
    assert issubclass(ConsensusEvidenceError, ConsensusPluginError)
    assert issubclass(ConsensusRejectedError, ConsensusPluginError)


def test_evidence_dict_is_immutable_snapshot_from_dataclass() -> None:
    evidence = build_plugin_evidence(
        plugin_id=DETERMINISTIC_TEST_ADAPTER_PLUGIN_ID,
        guarantee=GUARANTEE_MAJORITY_APPROVAL,
        state_id=STATE_ID,
        proposal_cid=PROPOSAL_CID,
        members=PEERS,
        approvals=("did:key:peer-a", "did:key:peer-b"),
    )
    payload = evidence.to_dict()
    mutated = copy.deepcopy(payload)
    mutated["guarantee"] = GUARANTEE_BFT
    # Original evidence object remains honest.
    assert evidence.guarantee == GUARANTEE_MAJORITY_APPROVAL
    # Mutated mapping still fails validation when re-checked as neighborhood.
    mutated["evidence_kind"] = "neighborhood"
    mutated["source"] = "profile_g_neighborhood"
    with pytest.raises(NeighborhoodGuaranteeError):
        validate_plugin_evidence(mutated)
