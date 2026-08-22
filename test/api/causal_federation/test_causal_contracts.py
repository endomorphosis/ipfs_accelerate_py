from __future__ import annotations

from dataclasses import replace

import pytest
from ipfs_accelerate_py.agent_supervisor.federation import contracts
from test.api.causal_federation.test_contracts import sample_contract


@pytest.mark.parametrize(
    "evidence_kind",
    [
        kind
        for kind in contracts.CausalEvidenceKind
        if kind is not contracts.CausalEvidenceKind.RETRIEVAL_NOMINATION
    ],
)
def test_declared_exact_evidence_kinds_may_be_authoritative(
    evidence_kind: contracts.CausalEvidenceKind,
) -> None:
    evidence = sample_contract(contracts.CausalEvidence)
    assert isinstance(evidence, contracts.CausalEvidence)

    exact = replace(
        evidence,
        evidence_kind=evidence_kind,
        authoritative=True,
    )

    assert exact.authoritative
    assert exact.evidence_kind is evidence_kind


def test_retrieval_nomination_cannot_manufacture_causal_authority() -> None:
    evidence = sample_contract(contracts.CausalEvidence)
    assert isinstance(evidence, contracts.CausalEvidence)

    with pytest.raises(contracts.FederationAuthorityError):
        replace(
            evidence,
            evidence_kind=contracts.CausalEvidenceKind.RETRIEVAL_NOMINATION,
            authoritative=True,
        )

    nomination = replace(
        evidence,
        evidence_kind=contracts.CausalEvidenceKind.RETRIEVAL_NOMINATION,
        authoritative=False,
    )
    assert not nomination.authoritative


@pytest.mark.parametrize(
    "status",
    [
        contracts.AbstractionFaithfulness.EMPIRICALLY_SUPPORTED,
        contracts.AbstractionFaithfulness.HEURISTIC,
        contracts.AbstractionFaithfulness.REFUTED,
        contracts.AbstractionFaithfulness.UNKNOWN,
    ],
)
def test_nomination_only_abstraction_status_cannot_be_policy_admitted(
    status: contracts.AbstractionFaithfulness,
) -> None:
    abstraction = sample_contract(contracts.CausalAbstractionMap)
    assert isinstance(abstraction, contracts.CausalAbstractionMap)

    with pytest.raises(contracts.FederationAuthorityError):
        replace(
            abstraction,
            faithfulness_status=status,
            policy_admitted=True,
        )


def test_conservative_abstraction_preserves_separate_policy_admission() -> None:
    abstraction = sample_contract(contracts.CausalAbstractionMap)
    assert isinstance(abstraction, contracts.CausalAbstractionMap)

    nomination = replace(
        abstraction,
        faithfulness_status=contracts.AbstractionFaithfulness.CONSERVATIVE,
        policy_admitted=False,
    )
    admitted = replace(nomination, policy_admitted=True)

    assert not nomination.policy_admitted
    assert admitted.policy_admitted
    assert admitted.faithfulness_status is contracts.AbstractionFaithfulness.CONSERVATIVE


def test_causal_edge_rejects_implicit_self_cycle() -> None:
    edge = sample_contract(contracts.CausalEdge)
    assert isinstance(edge, contracts.CausalEdge)

    with pytest.raises(contracts.FederationContractError):
        replace(edge, target_node_id=edge.source_node_id)


def test_causal_edge_requires_evidence() -> None:
    edge = sample_contract(contracts.CausalEdge)
    assert isinstance(edge, contracts.CausalEdge)

    with pytest.raises(contracts.FederationContractError):
        replace(edge, evidence_refs=())


@pytest.mark.parametrize("disposition", tuple(contracts.FrontierDisposition))
def test_frontier_disposition_round_trip(
    disposition: contracts.FrontierDisposition,
) -> None:
    entry = sample_contract(contracts.CausalFrontierEntry)
    assert isinstance(entry, contracts.CausalFrontierEntry)
    entry = replace(entry, disposition=disposition)

    decoded = contracts.CausalFrontierEntry.from_dict(entry.to_dict())

    assert decoded == entry
    assert decoded.disposition is disposition


def test_intervention_mismatch_is_explicit_and_content_addressed() -> None:
    intervention = sample_contract(contracts.InterventionTest)
    assert isinstance(intervention, contracts.InterventionTest)
    mismatch = replace(
        intervention,
        outcome="mismatched",
        mismatch_ref="counterexample:test",
    )

    decoded = contracts.InterventionTest.from_dict(mismatch.to_dict())

    assert decoded.mismatch_ref == "counterexample:test"
    assert decoded.outcome == "mismatched"
    assert decoded.cid == mismatch.cid


def test_unknown_causal_vocabulary_fails_closed() -> None:
    payload = sample_contract(contracts.CausalEdge).to_dict()
    payload["edge_kind"] = "SIMILAR_TO"

    with pytest.raises(ValueError):
        contracts.CausalEdge.from_dict(payload)
