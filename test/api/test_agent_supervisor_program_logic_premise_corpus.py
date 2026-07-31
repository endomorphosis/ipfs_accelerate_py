"""Tests for the independent content-addressed program premise corpus (LPR-005)."""

from __future__ import annotations

import sys

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.program_logic_prediction_contracts import (
    ProgramLogicAuthorityRoots,
)
from ipfs_accelerate_py.agent_supervisor.analysis.program_logic_premise_corpus import (
    ConsistencyDisposition,
    ConflictProofKind,
    DuplicatePremiseIdentityError,
    ForgedPremiseIdentityError,
    PremiseAuthority,
    PremiseAuthorityError,
    PremiseConflictProofError,
    PremiseConflictReceipt,
    PremiseConsistencyObligation,
    PremiseDependencyEdge,
    PremiseDerivationCycleError,
    PremiseEdgeKind,
    PremiseFeatureSet,
    PremiseLicensePolicy,
    PremiseSelfValidationError,
    PremiseSourceClass,
    PremiseSpanDigest,
    PremiseStructuralConflictError,
    PremiseTombstone,
    PremiseUnloweredDirectiveError,
    ProgramLogicPremise,
    ProgramLogicPremiseCorpus,
    ProgramLogicPremiseCorpusBuilder,
    ProgramLogicPremiseCorpusError,
    build_program_logic_premise_corpus,
    is_expectation_source_class,
    is_hypothesis_source_class,
    project_lazy_corpus_manifest,
)


@pytest.fixture
def roots() -> ProgramLogicAuthorityRoots:
    return ProgramLogicAuthorityRoots(
        repository_id="repository:one",
        objective_id="objective:one",
        trace_id="trace:one",
        change_id="change:one",
        consumer_id="consumer:one",
        forest_id="forest:one",
        tree_id="tree:one",
        overlay_id="overlay:one",
        graph_id="graph:one",
        index_id="index:one",
        corpus_id="corpus:one",
        model_id="model:one",
        translator_id="translator:one",
        toolchain_id="toolchain:one",
        policy_id="policy:one",
        environment_id="environment:one",
    )


def _features() -> PremiseFeatureSet:
    return PremiseFeatureSet(
        symbol_feature_refs=("symbol:process",),
        type_feature_refs=("type:Context",),
        effect_feature_refs=("effect:io",),
        import_feature_refs=("import:service.context",),
    )


def _span() -> PremiseSpanDigest:
    return PremiseSpanDigest(
        path="src/service.py",
        start_offset=10,
        end_offset=40,
        content_digest="sha256:" + "ab" * 32,
    )


def _license() -> PremiseLicensePolicy:
    return PremiseLicensePolicy(
        license_id="license:spdx:Apache-2.0",
        redaction_policy="span_only",
        export_policy="exportable",
    )


def _expectation(
    roots: ProgramLogicAuthorityRoots,
    premise_id: str = "p:spec.emit.tenant",
    *,
    statement_ref: str = "stmt:spec.emit.tenant",
    conflicts_with: tuple[str, ...] = (),
    dependency_edges: tuple[PremiseDependencyEdge, ...] = (),
) -> ProgramLogicPremise:
    return ProgramLogicPremise(
        roots=roots,
        premise_id=premise_id,
        source_class=PremiseSourceClass.REVIEWED_CONTRACT,
        statement_ref=statement_ref,
        statement_digest="sha256:" + ("11" * 32),
        lowering_ref="lower:contract:emit",
        expectation_authority=True,
        source_precedence=100,
        features=_features(),
        span=_span(),
        dependency_edges=dependency_edges,
        translation_refs=("translation:logic-ir@1",),
        assumption_refs=("assumption:stable-api",),
        invalidator_refs=("invalidate:tree-drift",),
        contract_identity="contract:ProgramContract@1",
        graph_identity=roots.graph_id,
        license_policy=_license(),
        conflicts_with=conflicts_with,
    )


def _hypothesis(
    roots: ProgramLogicAuthorityRoots,
    premise_id: str = "p:vector.decoy",
    *,
    source_class: PremiseSourceClass = PremiseSourceClass.VECTOR_ANALOGUE,
    statement_ref: str = "stmt:vector.decoy",
    dependency_edges: tuple[PremiseDependencyEdge, ...] = (),
) -> ProgramLogicPremise:
    return ProgramLogicPremise(
        roots=roots,
        premise_id=premise_id,
        source_class=source_class,
        statement_ref=statement_ref,
        statement_digest="sha256:" + ("22" * 32),
        lowering_ref="lower:hyp:vector",
        expectation_authority=False,
        source_precedence=10,
        features=_features(),
        license_policy=_license(),
        dependency_edges=dependency_edges,
    )


def test_expectation_and_hypothesis_source_classes_are_partitioned() -> None:
    assert is_expectation_source_class(PremiseSourceClass.REVIEWED_CONTRACT)
    assert is_expectation_source_class(PremiseSourceClass.NORMATIVE_SPEC)
    assert is_expectation_source_class(PremiseSourceClass.REVIEWED_CONFORMANCE_TEST)
    assert is_hypothesis_source_class(PremiseSourceClass.CANDIDATE_IMPLEMENTATION)
    assert is_hypothesis_source_class(PremiseSourceClass.COMMENT)
    assert is_hypothesis_source_class(PremiseSourceClass.RUNTIME_WITNESS)
    assert is_hypothesis_source_class(PremiseSourceClass.HISTORY)
    assert is_hypothesis_source_class(PremiseSourceClass.VECTOR_ANALOGUE)
    assert is_hypothesis_source_class(PremiseSourceClass.KNOWLEDGE_GRAPH)
    assert is_hypothesis_source_class(PremiseSourceClass.MODEL_HYPOTHESIS)
    assert not is_expectation_source_class(PremiseSourceClass.VECTOR_ANALOGUE)
    assert not is_hypothesis_source_class(PremiseSourceClass.REVIEWED_CONTRACT)


def test_builder_projects_expectations_and_hypotheses(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    builder = ProgramLogicPremiseCorpusBuilder(roots)
    builder.add_expectation(
        premise_id="p:spec.emit.tenant",
        source_class=PremiseSourceClass.REVIEWED_CONTRACT,
        statement_ref="stmt:spec.emit.tenant",
        lowering_ref="lower:contract:emit",
        features=_features(),
        span=_span(),
        translation_refs=("translation:logic-ir@1",),
        license_policy=_license(),
        contract_identity="contract:ProgramContract@1",
    )
    builder.add_expectation(
        premise_id="p:test.conformance",
        source_class=PremiseSourceClass.REVIEWED_CONFORMANCE_TEST,
        statement_ref="stmt:test.conformance",
        lowering_ref="lower:test:conformance",
    )
    builder.add_static_fact(
        premise_id="p:df.tenant_id.local",
        source_class=PremiseSourceClass.VALUE_PROVENANCE,
        statement_ref="stmt:df.tenant_id.local",
        lowering_ref="lower:df:tenant",
        features=_features(),
    )
    builder.add_hypothesis(
        premise_id="p:vector.decoy",
        source_class=PremiseSourceClass.VECTOR_ANALOGUE,
        statement_ref="stmt:vector.decoy",
        lowering_ref="lower:hyp:vector",
    )
    builder.add_hypothesis(
        premise_id="p:comment.prompt",
        source_class=PremiseSourceClass.COMMENT,
        statement_ref="stmt:comment.prompt",
        lowering_ref="lower:hyp:comment",
    )
    builder.add_hypothesis(
        premise_id="p:model.guess",
        source_class=PremiseSourceClass.MODEL_HYPOTHESIS,
        statement_ref="stmt:model.guess",
        lowering_ref="lower:hyp:model",
    )
    corpus = builder.build()

    assert len(corpus.premises) == 6
    assert len(corpus.expectation_premises()) == 2
    assert len(corpus.hypothesis_premises()) == 3
    assert all(not p.semantic_authority for p in corpus.premises)
    assert all(
        p.expectation_authority for p in corpus.expectation_premises()
    )
    assert all(
        not p.expectation_authority for p in corpus.hypothesis_premises()
    )
    assert corpus.consistency_disposition is ConsistencyDisposition.STRUCTURAL_INTEGRITY_OK
    assert corpus.content_id.startswith("b")
    assert ProgramLogicPremiseCorpus.from_dict(corpus.to_record()) == corpus


def test_premise_binds_roots_features_edges_span_translation_license(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    edge = PremiseDependencyEdge(
        from_premise_id="p:spec.emit.tenant",
        to_premise_id="p:df.tenant_id.local",
        kind=PremiseEdgeKind.ASSUMES,
    )
    premise = ProgramLogicPremise(
        roots=roots,
        premise_id="p:spec.emit.tenant",
        source_class=PremiseSourceClass.NORMATIVE_SPEC,
        statement_ref="stmt:spec.emit.tenant",
        statement_digest="sha256:" + ("33" * 32),
        lowering_ref="lower:spec:emit",
        expectation_authority=True,
        features=_features(),
        span=_span(),
        dependency_edges=(edge,),
        translation_refs=("translation:logic-ir@1", "translation:native@1"),
        assumption_refs=("assumption:stable-api",),
        invalidator_refs=("invalidate:tree-drift",),
        contract_identity="contract:RequiredBehavior@1",
        graph_identity=roots.graph_id,
        license_policy=_license(),
    )
    assert premise.roots.tree_id == "tree:one"
    assert premise.features is not None
    assert premise.features.symbol_feature_refs == ("symbol:process",)
    assert premise.features.type_feature_refs == ("type:Context",)
    assert premise.features.effect_feature_refs == ("effect:io",)
    assert premise.features.import_feature_refs == ("import:service.context",)
    assert premise.span is not None
    assert premise.span.path == "src/service.py"
    assert premise.translation_refs == (
        "translation:logic-ir@1",
        "translation:native@1",
    )
    assert premise.license_policy is not None
    assert premise.license_policy.redaction_policy == "span_only"
    assert premise.license_policy.export_policy == "exportable"
    assert premise.authority is PremiseAuthority.EXPECTATION
    assert premise.tree_identity == roots.tree_id
    assert ProgramLogicPremise.from_dict(premise.to_record()) == premise


def test_bodies_secrets_and_unlowered_directives_rejected(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    payload = _expectation(roots).to_record()
    payload["source_body"] = "def evil(): pass"
    with pytest.raises(
        ProgramLogicPremiseCorpusError, match="unsupported fields|source bodies"
    ):
        ProgramLogicPremise.from_dict(payload)

    # Nested body marker inside an allowed nested mapping must also fail closed.
    poisoned_roots = dict(roots.to_dict())
    poisoned_roots["source_body"] = "def evil(): pass"
    payload = _expectation(roots).to_record()
    payload["roots"] = poisoned_roots
    with pytest.raises(ProgramLogicPremiseCorpusError, match="source bodies"):
        ProgramLogicPremise.from_dict(payload)

    payload = _expectation(roots).to_record()
    payload["api_key"] = "sk-test"
    with pytest.raises(
        ProgramLogicPremiseCorpusError, match="unsupported fields|secret"
    ):
        ProgramLogicPremise.from_dict(payload)

    with pytest.raises(ProgramLogicPremiseCorpusError, match="secret"):
        ProgramLogicPremise(
            roots=roots,
            premise_id="p:secret",
            source_class=PremiseSourceClass.COMMENT,
            statement_ref="stmt:password=hunter2",
            statement_digest="sha256:" + ("55" * 32),
            lowering_ref="lower:secret",
        )

    with pytest.raises(PremiseUnloweredDirectiveError):
        ProgramLogicPremise(
            roots=roots,
            premise_id="p:unlowered",
            source_class=PremiseSourceClass.COMMENT,
            statement_ref="stmt:#unlowered directive remains",
            statement_digest="sha256:" + ("56" * 32),
            lowering_ref="lower:unlowered",
        )


def test_forged_and_duplicate_identities_rejected(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    premise = _expectation(roots)
    forged = premise.to_record()
    forged["content_id"] = "baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    with pytest.raises(ForgedPremiseIdentityError):
        ProgramLogicPremise.from_dict(forged)

    a = _expectation(roots, premise_id="p:same", statement_ref="stmt:a")
    b = ProgramLogicPremise(
        roots=roots,
        premise_id="p:same",
        source_class=PremiseSourceClass.REVIEWED_CONTRACT,
        statement_ref="stmt:b",
        statement_digest="sha256:" + ("99" * 32),
        lowering_ref="lower:b",
        expectation_authority=True,
    )
    with pytest.raises(DuplicatePremiseIdentityError):
        ProgramLogicPremiseCorpus(roots=roots, premises=(a, b))

    builder = ProgramLogicPremiseCorpusBuilder(roots)
    builder.add_expectation(
        premise_id="p:same",
        source_class=PremiseSourceClass.REVIEWED_CONTRACT,
        statement_ref="stmt:a",
        lowering_ref="lower:a",
        statement_digest="sha256:" + ("11" * 32),
    )
    with pytest.raises(DuplicatePremiseIdentityError):
        builder.add_expectation(
            premise_id="p:same",
            source_class=PremiseSourceClass.REVIEWED_CONTRACT,
            statement_ref="stmt:b",
            lowering_ref="lower:b",
            statement_digest="sha256:" + ("99" * 32),
        )


def test_self_validation_and_derivation_cycles_rejected(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    with pytest.raises(PremiseSelfValidationError):
        ProgramLogicPremise(
            roots=roots,
            premise_id="p:self",
            source_class=PremiseSourceClass.RUNTIME_WITNESS,
            statement_ref="stmt:self",
            statement_digest="sha256:" + ("66" * 32),
            lowering_ref="lower:self",
            self_validation=True,
        )

    with pytest.raises(PremiseSelfValidationError):
        ProgramLogicPremise(
            roots=roots,
            premise_id="p:self",
            source_class=PremiseSourceClass.RUNTIME_WITNESS,
            statement_ref="stmt:self",
            statement_digest="sha256:" + ("66" * 32),
            lowering_ref="lower:self",
            assumption_refs=("p:self",),
        )

    edge_ab = PremiseDependencyEdge(
        from_premise_id="p:cycle.a",
        to_premise_id="p:cycle.b",
        kind=PremiseEdgeKind.DERIVES_FROM,
    )
    edge_ba = PremiseDependencyEdge(
        from_premise_id="p:cycle.b",
        to_premise_id="p:cycle.a",
        kind=PremiseEdgeKind.DERIVES_FROM,
    )
    a = _hypothesis(
        roots,
        premise_id="p:cycle.a",
        source_class=PremiseSourceClass.THEOREM_CORPUS,
        statement_ref="stmt:cycle.a",
        dependency_edges=(edge_ab,),
    )
    b = ProgramLogicPremise(
        roots=roots,
        premise_id="p:cycle.b",
        source_class=PremiseSourceClass.THEOREM_CORPUS,
        statement_ref="stmt:cycle.b",
        statement_digest="sha256:" + ("77" * 32),
        lowering_ref="lower:cycle.b",
        dependency_edges=(edge_ba,),
    )
    with pytest.raises(PremiseDerivationCycleError):
        ProgramLogicPremiseCorpus(roots=roots, premises=(a, b))


def test_hypothesis_cannot_claim_expectation_or_semantic_authority(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    with pytest.raises(PremiseAuthorityError):
        ProgramLogicPremise(
            roots=roots,
            premise_id="p:bad-expect",
            source_class=PremiseSourceClass.VECTOR_ANALOGUE,
            statement_ref="stmt:bad",
            statement_digest="sha256:" + ("88" * 32),
            lowering_ref="lower:bad",
            expectation_authority=True,
        )

    with pytest.raises(PremiseAuthorityError):
        ProgramLogicPremise(
            roots=roots,
            premise_id="p:bad-sem",
            source_class=PremiseSourceClass.REVIEWED_CONTRACT,
            statement_ref="stmt:bad",
            statement_digest="sha256:" + ("88" * 32),
            lowering_ref="lower:bad",
            expectation_authority=True,
            semantic_authority=True,
        )

    with pytest.raises(PremiseAuthorityError):
        ProgramLogicPremiseCorpusBuilder(roots).add_hypothesis(
            premise_id="p:x",
            source_class=PremiseSourceClass.REVIEWED_CONTRACT,
            statement_ref="stmt:x",
            lowering_ref="lower:x",
        )


def test_structural_integrity_distinct_from_logical_consistency(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    """Corpus establishes structural integrity; logical consistency stays unknown
    unless a conflict receipt is independently replayed."""
    corpus = ProgramLogicPremiseCorpus(
        roots=roots,
        premises=(_expectation(roots), _hypothesis(roots)),
    )
    assert corpus.consistency_disposition is ConsistencyDisposition.STRUCTURAL_INTEGRITY_OK
    assert not corpus.conflict_receipts
    # Suspected authoritative contradiction emits obligations, not conflict receipts.
    builder = ProgramLogicPremiseCorpusBuilder(roots)
    builder.add_expectation(
        premise_id="p:spec.ms",
        source_class=PremiseSourceClass.REVIEWED_CONTRACT,
        statement_ref="stmt:timeout.ms",
        lowering_ref="lower:ms",
        conflicts_with=("p:spec.seconds",),
    )
    builder.add_expectation(
        premise_id="p:spec.seconds",
        source_class=PremiseSourceClass.REVIEWED_CONTRACT,
        statement_ref="stmt:timeout.seconds",
        lowering_ref="lower:seconds",
        conflicts_with=("p:spec.ms",),
    )
    corpus = builder.build()
    assert corpus.consistency_disposition is (
        ConsistencyDisposition.CONSISTENCY_OBLIGATION_EMITTED
    )
    assert corpus.consistency_obligations
    assert not corpus.conflict_receipts
    assert all(
        item.disposition
        in {
            ConsistencyDisposition.CONSISTENCY_OBLIGATION_EMITTED,
            ConsistencyDisposition.SUSPECTED_AUTHORITATIVE_CONTRADICTION,
        }
        for item in corpus.consistency_obligations
    )


def test_conflict_receipt_requires_independent_replay(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    with pytest.raises(PremiseConflictProofError):
        PremiseConflictReceipt(
            roots=roots,
            receipt_id="receipt:conflict:1",
            premise_ids=("p:spec.ms", "p:spec.seconds"),
            proof_kind=ConflictProofKind.UNSAT_CORE,
            proof_artifact_ref="proof:unsat:1",
            replay_receipt_ref="replay:1",
            translator_id=roots.translator_id,
            toolchain_id=roots.toolchain_id,
            independently_replayed=False,
            unsat_core_refs=("core:clause:1",),
        )

    with pytest.raises(PremiseConflictProofError):
        PremiseConflictReceipt(
            roots=roots,
            receipt_id="receipt:conflict:2",
            premise_ids=("p:spec.ms", "p:spec.seconds"),
            proof_kind=ConflictProofKind.UNSAT_CORE,
            proof_artifact_ref="proof:unsat:1",
            replay_receipt_ref="replay:1",
            translator_id=roots.translator_id,
            toolchain_id=roots.toolchain_id,
            unsat_core_refs=(),
        )

    receipt = PremiseConflictReceipt(
        roots=roots,
        receipt_id="receipt:conflict:ok",
        premise_ids=("p:spec.ms", "p:spec.seconds"),
        proof_kind=ConflictProofKind.UNSAT_CORE,
        proof_artifact_ref="proof:unsat:1",
        replay_receipt_ref="replay:1",
        translator_id=roots.translator_id,
        toolchain_id=roots.toolchain_id,
        unsat_core_refs=("core:clause:1", "core:clause:2"),
    )
    builder = ProgramLogicPremiseCorpusBuilder(roots)
    builder.add_expectation(
        premise_id="p:spec.ms",
        source_class=PremiseSourceClass.REVIEWED_CONTRACT,
        statement_ref="stmt:timeout.ms",
        lowering_ref="lower:ms",
        conflicts_with=("p:spec.seconds",),
    )
    builder.add_expectation(
        premise_id="p:spec.seconds",
        source_class=PremiseSourceClass.REVIEWED_CONTRACT,
        statement_ref="stmt:timeout.seconds",
        lowering_ref="lower:seconds",
    )
    builder.add_conflict_receipt(receipt)
    corpus = builder.build()
    assert corpus.consistency_disposition is ConsistencyDisposition.LOGICAL_CONFLICT_PROVED
    assert len(corpus.conflict_receipts) == 1
    assert corpus.conflict_receipts[0].independently_replayed is True

    # Claiming logical conflict without a receipt fails closed.
    with pytest.raises(PremiseConflictProofError):
        ProgramLogicPremiseCorpus(
            roots=roots,
            premises=(_expectation(roots),),
            consistency_disposition=ConsistencyDisposition.LOGICAL_CONFLICT_PROVED,
        )


def test_unknown_consistency_abstains_without_conflict_claim(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    empty = ProgramLogicPremiseCorpus(roots=roots)
    assert empty.consistency_disposition is ConsistencyDisposition.UNKNOWN
    assert not empty.conflict_receipts
    assert not empty.consistency_obligations


def test_tree_identity_mismatch_is_structural_conflict(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    with pytest.raises(PremiseStructuralConflictError):
        ProgramLogicPremise(
            roots=roots,
            premise_id="p:stale",
            source_class=PremiseSourceClass.REVIEWED_CONTRACT,
            statement_ref="stmt:stale",
            statement_digest="sha256:" + ("aa" * 32),
            lowering_ref="lower:stale",
            expectation_authority=True,
            tree_identity="tree:other",
        )


def test_incremental_rebuild_equals_clean_rebuild_including_tombstones(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    first = (
        ProgramLogicPremiseCorpusBuilder(roots)
        .add_expectation(
            premise_id="p:spec.a",
            source_class=PremiseSourceClass.REVIEWED_CONTRACT,
            statement_ref="stmt:a",
            lowering_ref="lower:a",
            statement_digest="sha256:" + ("a1" * 32),
        )
        .add_hypothesis(
            premise_id="p:hist.b",
            source_class=PremiseSourceClass.HISTORY,
            statement_ref="stmt:b",
            lowering_ref="lower:b",
            statement_digest="sha256:" + ("b1" * 32),
        )
        .build()
    )
    assert len(first.premises) == 2

    # Incremental: drop history premise; retain expectation; auto-tombstone.
    incremental = (
        ProgramLogicPremiseCorpusBuilder(roots)
        .with_previous(first)
        .add_expectation(
            premise_id="p:spec.a",
            source_class=PremiseSourceClass.REVIEWED_CONTRACT,
            statement_ref="stmt:a",
            lowering_ref="lower:a",
            statement_digest="sha256:" + ("a1" * 32),
        )
        .build()
    )
    assert len(incremental.premises) == 1
    assert len(incremental.tombstones) == 1
    assert incremental.tombstones[0].premise_id == "p:hist.b"
    assert incremental.tombstones[0].reason == "premise_removed"

    # Clean rebuild with the same live premises and tombstones.
    clean = (
        ProgramLogicPremiseCorpusBuilder(roots)
        .add_expectation(
            premise_id="p:spec.a",
            source_class=PremiseSourceClass.REVIEWED_CONTRACT,
            statement_ref="stmt:a",
            lowering_ref="lower:a",
            statement_digest="sha256:" + ("a1" * 32),
        )
        .add_tombstone(
            premise_id="p:hist.b",
            statement_digest="sha256:" + ("b1" * 32),
            reason="premise_removed",
        )
        .build()
    )
    assert incremental.content_id == clean.content_id
    assert incremental.revision == clean.revision
    assert [t.to_dict() for t in incremental.tombstones] == [
        t.to_dict() for t in clean.tombstones
    ]


def test_lazy_corpus_manifest_projection_is_body_free_and_structural_only(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    corpus = (
        ProgramLogicPremiseCorpusBuilder(roots)
        .add_expectation(
            premise_id="p:spec.emit.tenant",
            source_class=PremiseSourceClass.REVIEWED_CONTRACT,
            statement_ref="stmt:spec.emit.tenant",
            lowering_ref="lower:contract:emit",
            translation_refs=("translation:logic-ir@1",),
            license_policy=_license(),
        )
        .add_hypothesis(
            premise_id="p:model.guess",
            source_class=PremiseSourceClass.MODEL_HYPOTHESIS,
            statement_ref="stmt:model.guess",
            lowering_ref="lower:hyp:model",
        )
        .build()
    )
    projection = project_lazy_corpus_manifest(corpus)
    assert projection["schema"].endswith("lazy-corpus-manifest-projection@1")
    assert projection["corpus_revision"] == corpus.revision
    assert projection["logical_consistency_claimed"] is False
    assert projection["structural_integrity_only"] is True
    assert len(projection["theorems"]) == 2
    assert all(item["semantic_authority"] is False for item in projection["theorems"])
    assert "body" not in json_keys(projection)
    assert "source_body" not in json_keys(projection)
    # Must not require datasets types.
    assert "CorpusManifest" not in str(type(projection))


def json_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(str(key))
            keys |= json_keys(child)
    elif isinstance(value, list):
        for child in value:
            keys |= json_keys(child)
    return keys


def test_analysis_remains_cold_importable_when_datasets_missing() -> None:
    """Importing the premise corpus must not pull optional datasets/Hammer."""
    blocked = {
        name
        for name in list(sys.modules)
        if name == "ipfs_datasets_py"
        or name.startswith("ipfs_datasets_py.")
    }
    # Module under test is already importable at collection time; ensure its
    # dependency closure never required datasets for the public surface.
    import ipfs_accelerate_py.agent_supervisor.analysis.program_logic_premise_corpus as mod

    for name in (
        "ipfs_datasets_py",
        "ipfs_datasets_py.logic",
        "ipfs_datasets_py.logic.hammers",
        "ipfs_datasets_py.logic.hammers.corpus",
        "ipfs_datasets_py.logic.tactician",
    ):
        # If datasets happens to be present in the environment, the module
        # still must not have imported it as a hard dependency of its load.
        assert name not in getattr(mod, "__dict__", {})
        assert not any(
            attr for attr in dir(mod) if "CorpusManifest" == attr and name in blocked
        )
    # Public API does not expose Hammer types.
    assert not hasattr(mod, "CorpusManifest")
    assert not hasattr(mod, "TheoremEntry")
    assert callable(mod.project_lazy_corpus_manifest)
    assert callable(mod.ProgramLogicPremiseCorpusBuilder)


def test_build_convenience_and_round_trip(roots: ProgramLogicAuthorityRoots) -> None:
    corpus = build_program_logic_premise_corpus(
        roots,
        expectations=[
            {
                "premise_id": "p:spec.emit.tenant",
                "source_class": PremiseSourceClass.REVIEWED_CONTRACT,
                "statement_ref": "stmt:spec.emit.tenant",
                "lowering_ref": "lower:contract:emit",
                "features": _features(),
                "license_policy": _license(),
            }
        ],
        static_facts=[
            {
                "premise_id": "p:type.ctx",
                "source_class": PremiseSourceClass.TYPE_AND_EFFECT_FACTS,
                "statement_ref": "stmt:type.ctx",
                "lowering_ref": "lower:type:ctx",
            }
        ],
        hypotheses=[
            {
                "premise_id": "p:kg.edge",
                "source_class": PremiseSourceClass.KNOWLEDGE_GRAPH,
                "statement_ref": "stmt:kg.edge",
                "lowering_ref": "lower:kg",
            },
            {
                "premise_id": "p:impl.candidate",
                "source_class": PremiseSourceClass.CANDIDATE_IMPLEMENTATION,
                "statement_ref": "stmt:impl.candidate",
                "lowering_ref": "lower:impl",
            },
            {
                "premise_id": "p:runtime.w",
                "source_class": PremiseSourceClass.RUNTIME_WITNESS,
                "statement_ref": "stmt:runtime.w",
                "lowering_ref": "lower:runtime",
            },
        ],
    )
    assert len(corpus.premises) == 5
    restored = ProgramLogicPremiseCorpus.from_dict(corpus.to_record())
    assert restored == corpus
    assert restored.corpus_id == corpus.content_id


def test_native_conflict_proof_receipt(roots: ProgramLogicAuthorityRoots) -> None:
    receipt = PremiseConflictReceipt(
        roots=roots,
        receipt_id="receipt:native:1",
        premise_ids=("p:a", "p:b"),
        proof_kind=ConflictProofKind.NATIVE_CONFLICT_PROOF,
        proof_artifact_ref="proof:native:1",
        replay_receipt_ref="replay:native:1",
        translator_id=roots.translator_id,
        toolchain_id=roots.toolchain_id,
        native_conflict_refs=("native:thm:conflict",),
    )
    assert receipt.proof_kind is ConflictProofKind.NATIVE_CONFLICT_PROOF
    assert PremiseConflictReceipt.from_dict(receipt.to_record()) == receipt


def test_consistency_obligation_cannot_claim_logical_conflict(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    with pytest.raises(ProgramLogicPremiseCorpusError):
        PremiseConsistencyObligation(
            roots=roots,
            obligation_id="obligation:bad",
            premise_ids=("p:a", "p:b"),
            reason_code="suspected_authoritative_contradiction",
            disposition=ConsistencyDisposition.LOGICAL_CONFLICT_PROVED,
        )


def test_live_and_tombstoned_same_identity_is_structural_conflict(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    premise = _expectation(roots)
    tombstone = PremiseTombstone(
        premise_id=premise.premise_id,
        statement_digest=premise.statement_digest,
        reason="premise_removed",
        tree_identity=roots.tree_id,
    )
    with pytest.raises(PremiseStructuralConflictError):
        ProgramLogicPremiseCorpus(
            roots=roots,
            premises=(premise,),
            tombstones=(tombstone,),
        )
