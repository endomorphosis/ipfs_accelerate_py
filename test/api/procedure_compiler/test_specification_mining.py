from __future__ import annotations

import pytest
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.contracts import (
    ArtifactBindings,
    ArtifactState,
    ConditionOperator,
    EffectClass,
    IdempotencyClass,
    ProcedureBoundsError,
    ProcedureContractError,
)
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.invariant_mining import (
    InvariantCandidate,
    InvariantMiner,
    InvariantMiningError,
)
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.specification_mining import (
    MAX_MINING_SOURCES,
    PropertyKind,
    PropertyProposal,
    SpecificationCandidate,
    SpecificationMiner,
    SpecificationMiningError,
    SpecificationSource,
    SpecificationSourceKind,
    SpecificationTier,
    parse_specification_source,
)


def _bindings(*, tree: str = "tree-abc123", commit: str = "abc123") -> ArtifactBindings:
    return ArtifactBindings(
        repository_id="repo-main",
        repository_commit=commit,
        tree_id=tree,
        objective_id="PCPC-G000",
        task_id="PCPC-013",
        contract_revision="procedure-contracts-v1",
        policy_revision="authority-policy-v1",
        environment_id="python312-linux-lock1",
    )


def _source(
    source_kind: SpecificationSourceKind,
    source_cid: str,
    *,
    facts: dict[str, object] | None = None,
    proposals: tuple[PropertyProposal, ...] = (),
    authoritative: bool = False,
    producer_id: str = "source-producer@1",
    evidence_type: str = "admitted-source@1",
    bindings: ArtifactBindings | None = None,
) -> SpecificationSource:
    return SpecificationSource(
        bindings=bindings or _bindings(),
        source_kind=source_kind,
        source_cid=source_cid,
        producer_id=producer_id,
        evidence_type=evidence_type,
        admitted=True,
        authoritative=authoritative,
        proposals=proposals,
        facts=facts or {},
    )


def _by_kind(
    result_candidates: tuple[SpecificationCandidate, ...],
) -> dict[PropertyKind, SpecificationCandidate]:
    return {item.property_kind: item for item in result_candidates}


def test_miner_proposes_all_bounded_property_kinds_with_evidence_and_candidate_status() -> None:
    sources = (
        _source(
            SpecificationSourceKind.TYPES,
            "source-types",
            facts={"binding": "binding:tree_id", "operand": "tree-abc123"},
        ),
        _source(
            SpecificationSourceKind.TEST,
            "source-test",
            facts={"passing_test_count": 4, "operand": "admitted"},
        ),
        _source(
            SpecificationSourceKind.PROOF_OBLIGATION,
            "source-proof",
            facts={"binding": "binding:invariant", "operand": "scope-intact"},
        ),
        _source(
            SpecificationSourceKind.MUTANT,
            "source-mutant",
            facts={"unchanged_paths": ("src/untouched.py", "src/keep.py")},
        ),
        _source(
            SpecificationSourceKind.OPERATION_CONTRACT,
            "source-operation",
            facts={
                "effect_class": EffectClass.VALIDATION.value,
                "effect_id": "effect.validation",
                "targets": (),
                "idempotency_class": IdempotencyClass.IDEMPOTENT.value,
                "step_id": "step.tests",
                "step_ids": ("read", "tests", "postcondition"),
                "wall_time_ms": 60_000,
                "memory_bytes": 128_000_000,
            },
        ),
        _source(
            SpecificationSourceKind.FAILURE_SIGNATURE,
            "source-failure",
            facts={
                "rollback_id": "rollback.declared",
                "exact_target_cid": "rollback-target",
                "rollback_step_ids": ("rollback",),
                "trigger_effect_ids": ("effect.validation",),
            },
        ),
        _source(
            SpecificationSourceKind.RUNTIME_CHECK,
            "source-runtime",
            facts={
                "tree_id": "tree-abc123",
                "repository_commit": "abc123",
                "binding": "binding:runtime-invariant",
                "operand": "scope-intact",
            },
        ),
    )

    result = SpecificationMiner().mine(sources)
    kinds = _by_kind(result.candidates)

    assert set(kinds) == set(PropertyKind)
    for candidate in result.candidates:
        assert candidate.state is ArtifactState.CANDIDATE
        assert candidate.remains_candidate
        assert candidate.evidence_cids
        assert candidate.source_kinds
        assert all(item.property_id == candidate.property_id for item in candidate.evidence)
        assert all(item.tier in {SpecificationTier.CANDIDATE, SpecificationTier.NOMINATION} for item in candidate.evidence)

    assert kinds[PropertyKind.PRECONDITION].binding == "binding:tree_id"
    assert kinds[PropertyKind.POSTCONDITION].operator is ConditionOperator.ADMITTED
    assert kinds[PropertyKind.INVARIANT].operand == "scope-intact"
    assert kinds[PropertyKind.FRAME].operand == ("src/keep.py", "src/untouched.py")
    assert kinds[PropertyKind.EFFECT].operand == EffectClass.VALIDATION.value
    assert kinds[PropertyKind.RESOURCE].operand["wall_time_ms"] == 60_000
    assert kinds[PropertyKind.ORDER].operand == ("read", "tests", "postcondition")
    assert kinds[PropertyKind.IDEMPOTENCY].operand == IdempotencyClass.IDEMPOTENT.value
    assert kinds[PropertyKind.ROLLBACK].operand == "rollback-target"
    assert kinds[PropertyKind.FRESHNESS].operand == {
        "repository_commit": "abc123",
        "tree_id": "tree-abc123",
    }
    assert result.receipt.state is ArtifactState.CANDIDATE
    assert set(result.receipt.source_cids) == {source.source_cid for source in sources}
    assert result.invariant_candidates
    assert all(isinstance(item, InvariantCandidate) for item in result.invariant_candidates)
    assert all(item.state is ArtifactState.CANDIDATE for item in result.invariant_candidates)


def test_every_source_kind_retains_provenance_and_candidate_or_nomination_tier() -> None:
    sources = []
    for kind in SpecificationSourceKind:
        facts: dict[str, object] = {
            "property_kind": PropertyKind.PRECONDITION.value,
            "binding": "binding:tree_id",
            "operator": ConditionOperator.EXISTS.value,
            "operand": "tree-abc123",
        }
        if kind is SpecificationSourceKind.AUTHORITATIVE_DOCUMENTATION:
            sources.append(
                _source(
                    kind,
                    f"source-{kind.value}",
                    facts=facts,
                    authoritative=False,
                    producer_id="docs-producer@1",
                    evidence_type="design-note@1",
                )
            )
        else:
            sources.append(
                _source(
                    kind,
                    f"source-{kind.value}",
                    facts=facts,
                    producer_id=f"{kind.value}-producer@1",
                    evidence_type=f"{kind.value}-evidence@1",
                )
            )

    result = SpecificationMiner().mine(sources)
    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.state is ArtifactState.CANDIDATE
    assert {item.source_kind for item in candidate.evidence} == set(SpecificationSourceKind)
    by_kind = {item.source_kind: item for item in candidate.evidence}
    for kind in SpecificationSourceKind:
        evidence = by_kind[kind]
        assert evidence.source_cid == f"source-{kind.value}"
        assert evidence.producer_id == (
            "docs-producer@1"
            if kind is SpecificationSourceKind.AUTHORITATIVE_DOCUMENTATION
            else f"{kind.value}-producer@1"
        )
        assert evidence.admitted is True
        if kind is SpecificationSourceKind.AUTHORITATIVE_DOCUMENTATION:
            assert evidence.tier is SpecificationTier.NOMINATION
        else:
            assert evidence.tier is SpecificationTier.CANDIDATE
    assert candidate.tier is SpecificationTier.CANDIDATE


def test_frequency_absence_and_passing_tests_do_not_upgrade_candidate_status() -> None:
    frequent = _source(
        SpecificationSourceKind.ADMITTED_TRACE,
        "source-frequent",
        facts={
            "property_kind": PropertyKind.POSTCONDITION.value,
            "binding": "local:test-result",
            "operator": ConditionOperator.ADMITTED.value,
            "operand": "admitted",
            "occurrence_count": 64,
        },
    )
    passing = _source(
        SpecificationSourceKind.TEST,
        "source-passing",
        facts={
            "property_kind": PropertyKind.POSTCONDITION.value,
            "binding": "local:test-result",
            "operator": ConditionOperator.ADMITTED.value,
            "operand": "admitted",
            "passing_test_count": 18,
        },
    )
    absent = _source(
        SpecificationSourceKind.ADMITTED_TRACE,
        "source-absent",
        facts={
            "property_kind": PropertyKind.POSTCONDITION.value,
            "binding": "local:test-result",
            "operator": ConditionOperator.ADMITTED.value,
            "operand": "admitted",
            "occurrence_count": 0,
        },
    )

    result = SpecificationMiner().mine((frequent, passing, absent))
    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.state is ArtifactState.CANDIDATE
    assert candidate.supporting_count == 64
    assert candidate.passing_test_count == 18
    assert {item.source_cid for item in candidate.evidence} == {
        "source-frequent",
        "source-passing",
    }

    with pytest.raises(SpecificationMiningError, match="verified or promoted"):
        SpecificationCandidate(
            bindings=_bindings(),
            property_id=candidate.property_id,
            property_kind=candidate.property_kind,
            binding=candidate.binding,
            operator=candidate.operator,
            operand=candidate.operand,
            evidence=candidate.evidence,
            source_kinds=candidate.source_kinds,
            supporting_count=10_000,
            passing_test_count=10_000,
            state=ArtifactState.VERIFIED,
        )


def test_conflicting_evidence_yields_counterexample_and_refuses_the_property() -> None:
    first = _source(
        SpecificationSourceKind.ADMITTED_TRACE,
        "source-order-a",
        facts={"step_ids": ("read", "tests", "receipt")},
    )
    second = _source(
        SpecificationSourceKind.REJECTED_TRACE,
        "source-order-b",
        facts={
            "property_kind": PropertyKind.ORDER.value,
            "binding": "step.order",
            "operator": ConditionOperator.EQUALS.value,
            "operand": ("read", "receipt", "tests"),
        },
    )

    result = SpecificationMiner().mine((first, second))
    assert result.candidates == ()
    assert len(result.counterexamples) == 1
    counterexample = result.counterexamples[0]
    assert counterexample.violation_class == "conflicting_evidence"
    assert counterexample.state is ArtifactState.REJECTED
    assert set(counterexample.conflicting_source_cids) == {"source-order-a", "source-order-b"}
    assert len(counterexample.conflicting_operands) == 2
    assert counterexample.evidence_cids
    assert result.receipt.refused_property_ids == (counterexample.property_id,)
    assert result.receipt.candidate_cids == ()


def test_stale_freshness_is_refused_against_current_bindings() -> None:
    source = _source(
        SpecificationSourceKind.RUNTIME_CHECK,
        "source-stale",
        facts={
            "property_kind": PropertyKind.FRESHNESS.value,
            "binding": "binding:world_state",
            "tree_id": "other-tree",
            "repository_commit": "other-commit",
        },
    )

    result = SpecificationMiner().mine((source,))
    assert result.candidates == ()
    assert len(result.counterexamples) == 1
    assert result.counterexamples[0].violation_class == "stale_freshness"
    assert result.receipt.refusal_classes == ("stale_freshness",)


def test_unadmitted_and_foreign_sources_fail_closed() -> None:
    with pytest.raises(SpecificationMiningError, match="admitted"):
        SpecificationSource(
            bindings=_bindings(),
            source_kind=SpecificationSourceKind.TEST,
            source_cid="unadmitted",
            producer_id="tests@1",
            evidence_type="test@1",
            admitted=False,
        )

    with pytest.raises(SpecificationMiningError, match="authoritative"):
        _source(
            SpecificationSourceKind.TEST,
            "mislabelled",
            authoritative=True,
            facts={"binding": "local:test-result"},
        )

    native = _source(SpecificationSourceKind.TYPES, "native", facts={"binding": "binding:tree_id"})
    foreign = _source(
        SpecificationSourceKind.TYPES,
        "foreign",
        facts={"binding": "binding:tree_id"},
        bindings=_bindings(tree="other-tree"),
    )
    with pytest.raises(SpecificationMiningError, match="bindings differ"):
        SpecificationMiner().mine((native, foreign))


def test_source_and_candidate_round_trips_reject_unknown_fields_and_floats() -> None:
    source = _source(
        SpecificationSourceKind.OPERATION_CONTRACT,
        "source-round-trip",
        facts={
            "effect_class": EffectClass.PROOF.value,
            "effect_id": "effect.proof",
            "idempotency_class": IdempotencyClass.PURE.value,
            "step_id": "step.proof",
        },
    )
    assert parse_specification_source(source.to_dict()) == source
    assert parse_specification_source(source.to_json()) == source

    candidate = SpecificationMiner().mine((source,)).candidates[0]
    decoded = SpecificationCandidate.from_dict(candidate.to_dict())
    assert decoded == candidate
    assert decoded.content_id == candidate.content_id

    payload = source.to_dict()
    payload["claim_completion"] = True
    with pytest.raises(ProcedureContractError, match="unsupported fields"):
        parse_specification_source(payload)
    with pytest.raises(SpecificationMiningError, match="floating"):
        parse_specification_source('{"schema":"x","ratio":1.5}')


def test_mining_bounds_and_duplicate_source_provenance() -> None:
    with pytest.raises(SpecificationMiningError, match="at least one"):
        SpecificationMiner().mine(())
    with pytest.raises(ProcedureBoundsError):
        SpecificationMiner().mine(
            tuple(
                _source(
                    SpecificationSourceKind.TYPES,
                    f"source-{index}",
                    facts={"binding": "binding:tree_id"},
                )
                for index in range(MAX_MINING_SOURCES + 1)
            )
        )
    duplicate = _source(SpecificationSourceKind.TYPES, "same", facts={"binding": "binding:tree_id"})
    with pytest.raises(SpecificationMiningError, match="duplicate"):
        SpecificationMiner().mine((duplicate, duplicate))


def test_invariant_miner_keeps_candidate_status_and_refuses_conflicts() -> None:
    agreeing = _source(
        SpecificationSourceKind.PROOF_OBLIGATION,
        "inv-proof",
        facts={"binding": "binding:invariant", "operand": "scope-intact"},
    )
    runtime = _source(
        SpecificationSourceKind.RUNTIME_CHECK,
        "inv-runtime",
        facts={"binding": "binding:invariant", "operand": "scope-intact"},
    )
    mined = InvariantMiner().mine((agreeing, runtime))
    assert len(mined.candidates) == 1
    candidate = mined.candidates[0]
    assert candidate.property_kind is PropertyKind.INVARIANT
    assert candidate.state is ArtifactState.CANDIDATE
    assert candidate.supporting_count == 2
    assert {item.source_kind for item in candidate.evidence} == {
        SpecificationSourceKind.PROOF_OBLIGATION,
        SpecificationSourceKind.RUNTIME_CHECK,
    }
    assert InvariantCandidate.from_dict(candidate.to_dict()) == candidate

    conflicting = _source(
        SpecificationSourceKind.REJECTED_TRACE,
        "inv-conflict",
        facts={
            "property_kind": PropertyKind.INVARIANT.value,
            "binding": "binding:invariant",
            "operator": ConditionOperator.EQUALS.value,
            "operand": "scope-broken",
        },
    )
    refused = InvariantMiner().mine((agreeing, conflicting))
    assert refused.candidates == ()
    assert refused.refused_property_ids
    assert refused.counterexamples[0].violation_class == "conflicting_evidence"

    with pytest.raises(InvariantMiningError, match="candidate-tier"):
        InvariantCandidate(
            bindings=candidate.bindings,
            property_id=candidate.property_id,
            condition_id=candidate.condition_id,
            binding=candidate.binding,
            operator=candidate.operator,
            operand=candidate.operand,
            evidence=candidate.evidence,
            source_kinds=candidate.source_kinds,
            state=ArtifactState.PROMOTED,
        )


def test_non_authoritative_documentation_is_nomination_only() -> None:
    source = _source(
        SpecificationSourceKind.AUTHORITATIVE_DOCUMENTATION,
        "docs-only",
        authoritative=False,
        facts={
            "property_kind": PropertyKind.PRECONDITION.value,
            "binding": "binding:tree_id",
            "operator": ConditionOperator.CURRENT.value,
            "operand": "tree-abc123",
        },
    )
    result = SpecificationMiner().mine((source,))
    candidate = result.candidates[0]
    assert candidate.tier is SpecificationTier.NOMINATION
    assert candidate.state is ArtifactState.CANDIDATE
    assert candidate.evidence[0].tier is SpecificationTier.NOMINATION


def test_rejected_trace_and_documentation_must_name_a_claim() -> None:
    with pytest.raises(SpecificationMiningError, match="contradicted claim"):
        SpecificationMiner().mine((_source(SpecificationSourceKind.REJECTED_TRACE, "rejected-empty"),))
    with pytest.raises(SpecificationMiningError, match="bounded property"):
        SpecificationMiner().mine(
            (_source(SpecificationSourceKind.AUTHORITATIVE_DOCUMENTATION, "docs-empty"),)
        )
