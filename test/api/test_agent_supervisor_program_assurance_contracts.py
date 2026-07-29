from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from ipfs_accelerate_py.agent_supervisor.program_assurance_contracts import (
    MAX_CLAUSE_BYTES,
    ArtifactReference,
    AssuranceClaim,
    AssuranceLimits,
    AuthorityKind,
    ClaimLevel,
    ClaimPromotionError,
    ClaimVerdict,
    ContractBoundsError,
    ContractPrecedence,
    Counterexample,
    EvidenceFreshness,
    ExpectedContract,
    Finding,
    FindingSeverity,
    FindingStatus,
    ForgedIdentityError,
    InconclusiveState,
    ObservedContract,
    ProgramAssuranceContractError,
    RepositoryObservation,
    SemanticAuthorityError,
    StageReceipt,
    StageStatus,
    StaleAuthorityError,
    canonical_program_assurance_json_bytes,
    program_assurance_content_identity,
    validate_claim_promotion,
)


NOW = "2026-07-29T12:00:00Z"
EXPIRES = "2026-07-29T13:00:00Z"
SHA_A = "a" * 64
SHA_B = "b" * 64


def artifact(
    artifact_id: str = "artifact:source",
    *,
    kind: str = "source",
    digest: str = SHA_A,
) -> ArtifactReference:
    return ArtifactReference(
        artifact_id=artifact_id,
        kind=kind,
        sha256=digest,
        media_type="application/json",
        byte_count=123,
        uri=f"ipfs://{artifact_id}",
    )


def observation(*, expires: str = EXPIRES) -> RepositoryObservation:
    return RepositoryObservation(
        repository_id="repository:alpha",
        tree_id="tree:123",
        resolved_root="/srv/repositories/alpha",
        remote_id="https://example.invalid/alpha.git",
        commit_id="commit:abc",
        dirty=False,
        gitlink_tree_ids=("tree:dependency",),
        observed_at="2026-07-29T11:00:00Z",
        authority_expires_at=expires,
        analyzer_id="repository-observer",
        analyzer_version="1.2.3",
        policy_revision="policy:v1",
        artifacts=(artifact(),),
    )


def contracts(
    observed: RepositoryObservation,
) -> tuple[ExpectedContract, ObservedContract]:
    expected = ExpectedContract(
        repository_id=observed.repository_id,
        tree_id=observed.tree_id,
        symbol="pkg.api.call",
        interface="mcp://pkg/call",
        policy_revision=observed.policy_revision,
        precedence=ContractPrecedence.REVIEWED_INTERFACE,
        summary="Calls return a bounded success envelope.",
        clauses=("result.status == 'ok'", "result.body_bytes <= 4096"),
        source_artifact_ids=("artifact:source",),
    )
    actual = ObservedContract(
        repository_id=observed.repository_id,
        tree_id=observed.tree_id,
        symbol=expected.symbol,
        interface=expected.interface,
        policy_revision=observed.policy_revision,
        repository_observation_id=observed.observation_id,
        summary="A bound fixture returned an error envelope.",
        clauses=("result.status == 'error'",),
        source_artifact_ids=("artifact:runtime",),
    )
    return expected, actual


def broken_evidence() -> tuple[
    RepositoryObservation,
    ExpectedContract,
    ObservedContract,
    Counterexample,
    AssuranceClaim,
    Finding,
]:
    observed = observation()
    expected, actual = contracts(observed)
    witness = artifact("artifact:witness", kind="counterexample", digest=SHA_B)
    counterexample = Counterexample(
        repository_id=observed.repository_id,
        tree_id=observed.tree_id,
        symbol=expected.symbol,
        interface=expected.interface,
        policy_revision=observed.policy_revision,
        expected_contract_id=expected.expected_contract_id,
        observed_contract_id=actual.observed_contract_id,
        summary="The fixture produces an error where success is required.",
        witness_steps=("invoke fixture input 7", "observe status error"),
        artifacts=(witness,),
        evaluated_at=NOW,
        authority_expires_at=EXPIRES,
    )
    claim = AssuranceClaim(
        repository_id=observed.repository_id,
        tree_id=observed.tree_id,
        symbol=expected.symbol,
        interface=expected.interface,
        policy_revision=observed.policy_revision,
        repository_observation_id=observed.observation_id,
        claim_level=ClaimLevel.MODEL_DISPROVED,
        verdict=ClaimVerdict.VIOLATED,
        inconclusive_state=InconclusiveState.NONE,
        authority_kind=AuthorityKind.PROOF_KERNEL,
        producer_id="kernel:model-checker",
        producer_version="2.0",
        evaluated_at=NOW,
        authority_expires_at=EXPIRES,
        expected_contract_id=expected.expected_contract_id,
        observed_contract_id=actual.observed_contract_id,
        counterexample_id=counterexample.counterexample_id,
        assumptions=("fixture is hermetic",),
        artifacts=(witness,),
        confidence_millionths=1_000_000,
        semantic_proof=True,
    )
    finding = Finding(
        status=FindingStatus.CONTRACT_BROKEN,
        severity=FindingSeverity.HIGH,
        summary="The implementation violates its reviewed interface.",
        claim=claim,
        expected_contract=expected,
        observed_contract=actual,
        counterexample=counterexample,
        affected_paths=("src/api.py",),
        remediation_scope=("pkg.api.call",),
        artifacts=(witness,),
    )
    return observed, expected, actual, counterexample, claim, finding


def test_claim_level_vocabulary_is_exact_and_non_hierarchical() -> None:
    assert {item.value for item in ClaimLevel} == {
        "observed_syntax",
        "resolved_static",
        "model_proved",
        "model_disproved",
        "runtime_witnessed",
        "zk_trace_attested",
    }
    validate_claim_promotion(ClaimLevel.MODEL_PROVED, ClaimLevel.MODEL_PROVED)
    with pytest.raises(ClaimPromotionError):
        validate_claim_promotion(
            ClaimLevel.OBSERVED_SYNTAX, ClaimLevel.RESOLVED_STATIC
        )
    with pytest.raises(ClaimPromotionError):
        ClaimLevel.ZK_TRACE_ATTESTED.require(ClaimLevel.MODEL_PROVED)


def test_round_trip_identity_and_immutability_for_all_records() -> None:
    observed, expected, actual, counterexample, claim, finding = broken_evidence()
    limits = AssuranceLimits(max_claims=4, max_findings=4, max_artifacts=16)
    receipt = StageReceipt(
        stage="model-check",
        status=StageStatus.COMPLETED,
        claim_level=ClaimLevel.MODEL_DISPROVED,
        inconclusive_state=InconclusiveState.NONE,
        observation=observed,
        analyzer_id="model-checker",
        analyzer_version="2.0",
        objective_revision="objective:v1",
        policy_revision=observed.policy_revision,
        configuration_digest="sha256:" + SHA_A,
        evaluated_at=NOW,
        authority_expires_at=EXPIRES,
        coverage_complete=True,
        truncated=False,
        claims=(claim,),
        findings=(finding,),
        artifacts=(artifact("artifact:stage"),),
        limits=limits,
    )

    records = (
        artifact(),
        observed,
        expected,
        actual,
        counterexample,
        claim,
        finding,
        limits,
        receipt,
    )
    for record in records:
        restored = type(record).from_json(record.to_json())
        assert restored == record
        assert restored.content_id == record.content_id
        assert restored.canonical_bytes() == record.canonical_bytes()

    assert receipt.authoritative
    assert receipt.safe_for_semantic_reasoning
    assert finding.actionable
    with pytest.raises(FrozenInstanceError):
        receipt.stage = "forged"  # type: ignore[misc]


def test_serialization_is_deterministic_and_order_independent_where_declared() -> None:
    first = artifact("artifact:a", digest=SHA_A)
    second = artifact("artifact:b", digest=SHA_B)
    left = observation()
    right = RepositoryObservation(
        repository_id=left.repository_id,
        tree_id=left.tree_id,
        resolved_root=left.resolved_root,
        remote_id=left.remote_id,
        commit_id=left.commit_id,
        dirty=left.dirty,
        gitlink_tree_ids=left.gitlink_tree_ids,
        observed_at=left.observed_at,
        authority_expires_at=left.authority_expires_at,
        analyzer_id=left.analyzer_id,
        analyzer_version=left.analyzer_version,
        policy_revision=left.policy_revision,
        artifacts=(second, first),
    )
    reordered = RepositoryObservation(
        repository_id=right.repository_id,
        tree_id=right.tree_id,
        resolved_root=right.resolved_root,
        remote_id=right.remote_id,
        commit_id=right.commit_id,
        dirty=right.dirty,
        gitlink_tree_ids=right.gitlink_tree_ids,
        observed_at=right.observed_at,
        authority_expires_at=right.authority_expires_at,
        analyzer_id=right.analyzer_id,
        analyzer_version=right.analyzer_version,
        policy_revision=right.policy_revision,
        artifacts=(first, second),
    )
    assert right.canonical_bytes() == reordered.canonical_bytes()
    assert right.observation_id == reordered.observation_id

    payload_a = {"z": [1, 2], "a": {"y": True, "x": None}}
    payload_b = {"a": {"x": None, "y": True}, "z": [1, 2]}
    assert canonical_program_assurance_json_bytes(payload_a) == (
        canonical_program_assurance_json_bytes(payload_b)
    )
    assert program_assurance_content_identity(payload_a) == (
        program_assurance_content_identity(payload_b)
    )


@pytest.mark.parametrize(
    "record_factory,id_name",
    [
        (lambda: artifact(), "reference_id"),
        (lambda: observation(), "observation_id"),
        (lambda: broken_evidence()[1], "expected_contract_id"),
        (lambda: broken_evidence()[2], "observed_contract_id"),
        (lambda: broken_evidence()[3], "counterexample_id"),
        (lambda: broken_evidence()[4], "claim_id"),
        (lambda: broken_evidence()[5], "finding_id"),
    ],
)
def test_forged_record_identities_are_rejected(record_factory, id_name) -> None:
    record = record_factory()
    payload = record.to_record()
    payload[id_name] = "baguqeeraforged"
    with pytest.raises(ForgedIdentityError):
        type(record).from_dict(payload)


def test_nested_identity_forgery_is_rejected() -> None:
    finding = broken_evidence()[5]
    payload = finding.to_record()
    payload["expected_contract"]["summary"] = "attacker changed the expectation"
    with pytest.raises(ForgedIdentityError):
        Finding.from_dict(payload)


def test_bounds_and_unbounded_body_fields_are_rejected() -> None:
    with pytest.raises(ContractBoundsError):
        ExpectedContract(
            repository_id="repo",
            tree_id="tree",
            symbol="symbol",
            interface="interface",
            policy_revision="policy",
            precedence=ContractPrecedence.PUBLIC_SIGNATURE,
            summary="summary",
            clauses=("x" * (MAX_CLAUSE_BYTES + 1),),
            source_artifact_ids=("artifact",),
        )
    with pytest.raises(ContractBoundsError):
        AssuranceLimits(max_claims=0)

    payload = artifact().to_record()
    payload["body"] = "source text must never be embedded"
    with pytest.raises(ProgramAssuranceContractError):
        ArtifactReference.from_dict(payload)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_values_are_rejected(value: float) -> None:
    with pytest.raises(ValueError):
        canonical_program_assurance_json_bytes({"confidence": value})


def test_explicit_inconclusive_states_are_required() -> None:
    observed = observation()
    evidence = artifact("artifact:ambiguous")
    claim = AssuranceClaim(
        repository_id=observed.repository_id,
        tree_id=observed.tree_id,
        symbol="pkg.dynamic",
        interface="python",
        policy_revision=observed.policy_revision,
        repository_observation_id=observed.observation_id,
        claim_level=ClaimLevel.RESOLVED_STATIC,
        verdict=ClaimVerdict.INCONCLUSIVE,
        inconclusive_state=InconclusiveState.AMBIGUOUS,
        authority_kind=AuthorityKind.STATIC_RESOLVER,
        producer_id="resolver",
        producer_version="1",
        evaluated_at=NOW,
        authority_expires_at=EXPIRES,
        artifacts=(evidence,),
    )
    finding = Finding(
        status=FindingStatus.AMBIGUOUS,
        severity=FindingSeverity.INFO,
        summary="Dynamic dispatch has multiple candidate targets.",
        claim=claim,
    )
    receipt = StageReceipt(
        stage="resolve",
        status=StageStatus.INCONCLUSIVE,
        claim_level=ClaimLevel.RESOLVED_STATIC,
        inconclusive_state=InconclusiveState.AMBIGUOUS,
        observation=observed,
        analyzer_id="resolver",
        analyzer_version="1",
        objective_revision="objective:v1",
        policy_revision=observed.policy_revision,
        configuration_digest="config:v1",
        evaluated_at=NOW,
        authority_expires_at=EXPIRES,
        coverage_complete=False,
        truncated=False,
        claims=(claim,),
        findings=(finding,),
    )
    assert not claim.authoritative
    assert not receipt.authoritative
    assert StageReceipt.from_json(receipt.to_json()) == receipt

    with pytest.raises(ProgramAssuranceContractError):
        AssuranceClaim(
            **{
                **claim.__dict__,
                "verdict": ClaimVerdict.INCONCLUSIVE,
                "inconclusive_state": InconclusiveState.NONE,
            }
        )


def test_stale_authority_and_forged_authority_projection_are_rejected() -> None:
    observed = observation(expires="2026-07-29T11:30:00Z")
    stale_claim = AssuranceClaim(
        repository_id=observed.repository_id,
        tree_id=observed.tree_id,
        symbol="pkg.api.call",
        interface="mcp://pkg/call",
        policy_revision=observed.policy_revision,
        repository_observation_id=observed.observation_id,
        claim_level=ClaimLevel.OBSERVED_SYNTAX,
        verdict=ClaimVerdict.INCONCLUSIVE,
        inconclusive_state=InconclusiveState.STALE,
        authority_kind=AuthorityKind.PARSER,
        producer_id="parser",
        producer_version="1",
        evaluated_at=NOW,
        authority_expires_at="2026-07-29T11:45:00Z",
        artifacts=(artifact(),),
    )
    assert stale_claim.freshness is EvidenceFreshness.STALE
    payload = stale_claim.to_record()
    payload["authoritative"] = True
    with pytest.raises(StaleAuthorityError):
        AssuranceClaim.from_dict(payload)

    with pytest.raises(StaleAuthorityError):
        AssuranceClaim(
            repository_id=observed.repository_id,
            tree_id=observed.tree_id,
            symbol="pkg.api.call",
            interface="mcp://pkg/call",
            policy_revision=observed.policy_revision,
            repository_observation_id=observed.observation_id,
            claim_level=ClaimLevel.OBSERVED_SYNTAX,
            verdict=ClaimVerdict.SATISFIED,
            inconclusive_state=InconclusiveState.NONE,
            authority_kind=AuthorityKind.PARSER,
            producer_id="parser",
            producer_version="1",
            evaluated_at=NOW,
            authority_expires_at="2026-07-29T11:45:00Z",
            artifacts=(artifact(),),
        )


def test_illegal_promotion_and_wrong_authority_are_rejected() -> None:
    observed = observation()
    common = dict(
        repository_id=observed.repository_id,
        tree_id=observed.tree_id,
        symbol="pkg.api.call",
        interface="mcp://pkg/call",
        policy_revision=observed.policy_revision,
        repository_observation_id=observed.observation_id,
        verdict=ClaimVerdict.SATISFIED,
        inconclusive_state=InconclusiveState.NONE,
        producer_id="producer",
        producer_version="1",
        evaluated_at=NOW,
        authority_expires_at=EXPIRES,
        artifacts=(artifact(),),
    )
    with pytest.raises(ClaimPromotionError):
        AssuranceClaim(
            **common,
            claim_level=ClaimLevel.RESOLVED_STATIC,
            authority_kind=AuthorityKind.STATIC_RESOLVER,
            source_claim_level=ClaimLevel.OBSERVED_SYNTAX,
        )
    with pytest.raises(SemanticAuthorityError):
        AssuranceClaim(
            **common,
            claim_level=ClaimLevel.RUNTIME_WITNESSED,
            authority_kind=AuthorityKind.PARSER,
        )
    with pytest.raises(SemanticAuthorityError):
        AssuranceClaim(
            **common,
            claim_level=ClaimLevel.OBSERVED_SYNTAX,
            authority_kind=AuthorityKind.PARSER,
            semantic_proof=True,
        )


def test_zk_trace_attestation_is_never_semantic_proof() -> None:
    observed = observation()
    common = dict(
        repository_id=observed.repository_id,
        tree_id=observed.tree_id,
        symbol="trace:model-check",
        interface="circuit:v1",
        policy_revision=observed.policy_revision,
        repository_observation_id=observed.observation_id,
        claim_level=ClaimLevel.ZK_TRACE_ATTESTED,
        verdict=ClaimVerdict.SATISFIED,
        inconclusive_state=InconclusiveState.NONE,
        authority_kind=AuthorityKind.ZK_VERIFIER,
        producer_id="zk-verifier",
        producer_version="1",
        evaluated_at=NOW,
        authority_expires_at=EXPIRES,
        artifacts=(artifact("artifact:zk-receipt", kind="zk_receipt"),),
    )
    with pytest.raises(SemanticAuthorityError):
        AssuranceClaim(**common, semantic_proof=True)

    claim = AssuranceClaim(**common, semantic_proof=False)
    receipt = StageReceipt(
        stage="attest-trace",
        status=StageStatus.COMPLETED,
        claim_level=ClaimLevel.ZK_TRACE_ATTESTED,
        inconclusive_state=InconclusiveState.NONE,
        observation=observed,
        analyzer_id="zk-verifier",
        analyzer_version="1",
        objective_revision="objective:v1",
        policy_revision=observed.policy_revision,
        configuration_digest="circuit:v1",
        evaluated_at=NOW,
        authority_expires_at=EXPIRES,
        coverage_complete=True,
        truncated=False,
        claims=(claim,),
    )
    assert receipt.authoritative
    assert not receipt.safe_for_semantic_reasoning
    payload = receipt.to_record()
    payload["safe_for_semantic_reasoning"] = True
    with pytest.raises(ForgedIdentityError):
        StageReceipt.from_dict(payload)


def test_contract_broken_requires_exact_scope_and_conclusive_counterexample() -> None:
    observed, expected, actual, counterexample, claim, _ = broken_evidence()
    foreign_expected = ExpectedContract(
        repository_id=expected.repository_id,
        tree_id="tree:foreign",
        symbol=expected.symbol,
        interface=expected.interface,
        policy_revision=expected.policy_revision,
        precedence=expected.precedence,
        summary=expected.summary,
        clauses=expected.clauses,
        source_artifact_ids=expected.source_artifact_ids,
    )
    with pytest.raises(SemanticAuthorityError):
        Finding(
            status=FindingStatus.CONTRACT_BROKEN,
            severity=FindingSeverity.HIGH,
            summary="detached evidence",
            claim=claim,
            expected_contract=foreign_expected,
            observed_contract=actual,
            counterexample=counterexample,
        )

    zk_claim = AssuranceClaim(
        repository_id=observed.repository_id,
        tree_id=observed.tree_id,
        symbol=expected.symbol,
        interface=expected.interface,
        policy_revision=observed.policy_revision,
        repository_observation_id=observed.observation_id,
        claim_level=ClaimLevel.ZK_TRACE_ATTESTED,
        verdict=ClaimVerdict.VIOLATED,
        inconclusive_state=InconclusiveState.NONE,
        authority_kind=AuthorityKind.ZK_VERIFIER,
        producer_id="zk",
        producer_version="1",
        evaluated_at=NOW,
        authority_expires_at=EXPIRES,
        expected_contract_id=expected.expected_contract_id,
        observed_contract_id=actual.observed_contract_id,
        counterexample_id=counterexample.counterexample_id,
        artifacts=(artifact("artifact:zk"),),
        semantic_proof=False,
    )
    with pytest.raises(SemanticAuthorityError):
        Finding(
            status=FindingStatus.CONTRACT_BROKEN,
            severity=FindingSeverity.HIGH,
            summary="ZK receipts do not establish semantic violations.",
            claim=zk_claim,
            expected_contract=expected,
            observed_contract=actual,
            counterexample=counterexample,
        )
