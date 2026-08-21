"""Hermetic PCAR-004 semantic-entropy dimension and authority tests."""

from __future__ import annotations

import json

import pytest

from ipfs_accelerate_py.agent_supervisor.architecture_refactorer.architecture_ir import (
    ArchitectureEdge,
    ArchitectureIR,
    ArchitectureNode,
)
from ipfs_accelerate_py.agent_supervisor.architecture_refactorer.contracts import (
    Confidence,
    EdgeKind,
    NodeKind,
    SourceFactIdentity,
    SourceSpan,
)
from ipfs_accelerate_py.agent_supervisor.architecture_refactorer.entropy import (
    CHANGE_AMPLIFICATION_SCHEMA,
    CLOSED_ENTROPY_DIMENSIONS,
    CLOSED_NON_AUTHORITY_CLAIMS,
    CLOSED_SAFETY_PREDICATES,
    ENTROPY_DIMENSION_SCHEMA,
    ENTROPY_EXTRACTOR_IDENTITY,
    ENTROPY_IS_PRIORITIZATION_ONLY,
    ENTROPY_REPORT_EVIDENCE,
    FROZEN_TASK_CORPUS_SCHEMA,
    NON_COMPENSABLE_INVARIANTS,
    RANKING_IS_NON_PROBATIVE,
    REQUIRED_ENTROPY_DIMENSIONS,
    SEMANTIC_ENTROPY_EVIDENCE,
    SEMANTIC_ENTROPY_SCHEMA,
    SEMANTIC_ENTROPY_VERSION,
    ChangeAmplificationMeasure,
    EntropyAuthorityError,
    EntropyContractError,
    EntropyDimensionKind,
    EntropyDimensionRecord,
    FrozenTaskCorpus,
    SemanticEntropyReport,
    canonical_entropy_vector,
    derive_non_probative_ranking,
    entropy_establishes,
    entropy_satisfies_safety_predicate,
    measure_change_amplification,
    measure_entropy_dimensions,
    measure_semantic_entropy,
    refuse_entropy_authority,
)
from ipfs_accelerate_py.utils.cid_utils import cid_for_dag_json, validate_cid

_TREE = "a698da9e4b54e2929adacb613bc61ba3e72eed58"
_FRESHNESS = "pcar-004-fixture"
_EXTRACTOR = "pcar-004-fixture"
_CODE_PATH = "ipfs_accelerate_py/agent_supervisor/architecture_refactorer/entropy.py"
_DOC_PATH = "docs/architecture/guide.md"
_CORPUS = FrozenTaskCorpus(corpus_id="pcar-004-fixture-corpus", task_ids=("PCAR-004",))


def _span(path: str, start: int, end: int) -> SourceSpan:
    return SourceSpan(path, start, end)


def _fact(
    path: str,
    start: int,
    end: int,
    *,
    confidence: Confidence = Confidence.EXACT,
) -> SourceFactIdentity:
    return SourceFactIdentity(
        extractor_identity=_EXTRACTOR,
        span=_span(path, start, end),
        confidence=confidence,
        freshness=_FRESHNESS,
        repository_tree=_TREE,
    )


def _node(
    node_id: str,
    kind: NodeKind,
    *,
    start: int,
    path: str = _CODE_PATH,
    confidence: Confidence = Confidence.EXACT,
) -> ArchitectureNode:
    return ArchitectureNode(
        node_id=node_id,
        kind=kind,
        provenance=_fact(path, start, start, confidence=confidence),
    )


def _edge(
    edge_id: str,
    kind: EdgeKind,
    source: str,
    target: str,
    *,
    start: int,
    path: str = _CODE_PATH,
    confidence: Confidence = Confidence.EXACT,
) -> ArchitectureEdge:
    return ArchitectureEdge(
        edge_id=edge_id,
        kind=kind,
        source=source,
        target=target,
        provenance=_fact(path, start, start, confidence=confidence),
    )


def _graph(
    nodes: tuple[ArchitectureNode, ...],
    edges: tuple[ArchitectureEdge, ...] = (),
) -> ArchitectureIR:
    return ArchitectureIR.from_parts(
        repository_tree=_TREE,
        freshness=_FRESHNESS,
        nodes=nodes,
        edges=edges,
    )


def _clean_nodes() -> tuple[ArchitectureNode, ...]:
    return (
        _node("n-pkg", NodeKind.PACKAGE, start=1),
        _node("n-mod", NodeKind.MODULE, start=2),
        _node("n-file", NodeKind.FILE, start=3),
        _node("n-sym-a", NodeKind.SYMBOL, start=4),
        _node("n-sym-b", NodeKind.SYMBOL, start=5),
        _node("n-iface", NodeKind.INTERFACE, start=6),
        _node("n-schema", NodeKind.SCHEMA, start=7),
        _node("n-op", NodeKind.OPERATION, start=8),
        _node("n-effect", NodeKind.EFFECT, start=9),
        _node("n-auth", NodeKind.AUTHORITY, start=10),
        _node("n-policy", NodeKind.POLICY, start=11),
        _node("n-state", NodeKind.STATE, start=12),
        _node("n-receipt", NodeKind.RECEIPT, start=13),
        _node("n-test", NodeKind.TEST, start=14),
        _node("n-proof", NodeKind.PROOF, start=15),
        _node("n-prov", NodeKind.PROVIDER, start=16),
        _node("n-entry", NodeKind.ENTRYPOINT, start=17),
        _node("n-art", NodeKind.ARTIFACT, start=18, path=_DOC_PATH),
        _node("n-gen", NodeKind.GENERATED, start=19),
    )


def _clean_edges(*, calls_confidence: Confidence = Confidence.EXACT) -> tuple[ArchitectureEdge, ...]:
    return (
        _edge("e-contains-pkg", EdgeKind.CONTAINS, "n-pkg", "n-mod", start=1),
        _edge("e-contains-mod", EdgeKind.CONTAINS, "n-mod", "n-file", start=2),
        _edge("e-contains-sym-a", EdgeKind.CONTAINS, "n-file", "n-sym-a", start=3),
        _edge("e-contains-sym-b", EdgeKind.CONTAINS, "n-file", "n-sym-b", start=4),
        _edge("e-contains-effect", EdgeKind.CONTAINS, "n-file", "n-effect", start=5),
        _edge("e-contains-iface", EdgeKind.CONTAINS, "n-file", "n-iface", start=6),
        _edge("e-implements", EdgeKind.IMPLEMENTS, "n-entry", "n-iface", start=7),
        _edge("e-executes", EdgeKind.EXECUTES, "n-entry", "n-op", start=8),
        _edge("e-calls", EdgeKind.CALLS, "n-op", "n-sym-a", start=9, confidence=calls_confidence),
        _edge("e-auth-op", EdgeKind.AUTHORIZES, "n-auth", "n-op", start=10),
        _edge("e-auth-state", EdgeKind.AUTHORIZES, "n-auth", "n-state", start=11),
        _edge("e-persists", EdgeKind.PERSISTS, "n-op", "n-state", start=12),
        _edge("e-writes", EdgeKind.WRITES, "n-sym-a", "n-file", start=13),
        _edge("e-tests", EdgeKind.TESTS, "n-test", "n-op", start=14),
        _edge("e-proves", EdgeKind.PROVES, "n-proof", "n-op", start=15),
        _edge("e-generates", EdgeKind.GENERATES, "n-gen", "n-art", start=16),
        _edge("e-confirms", EdgeKind.CONFIRMS, "n-art", "n-schema", start=17, path=_DOC_PATH),
        _edge("e-reads", EdgeKind.READS, "n-op", "n-schema", start=18),
        _edge("e-policy", EdgeKind.EVALUATES_POLICY, "n-op", "n-policy", start=19),
        _edge("e-constructs", EdgeKind.CONSTRUCTS, "n-op", "n-effect", start=20),
    )


def _clean_graph(*, calls_confidence: Confidence = Confidence.EXACT) -> ArchitectureIR:
    return _graph(_clean_nodes(), _clean_edges(calls_confidence=calls_confidence))


def _noisy_graph() -> ArchitectureIR:
    extra_nodes = (
        _node("n-auth-b", NodeKind.AUTHORITY, start=30),
        _node("n-file-b", NodeKind.FILE, start=31),
        _node("n-schema-old", NodeKind.SCHEMA, start=32),
        _node("n-receipt-old", NodeKind.RECEIPT, start=33),
        _node("n-compat", NodeKind.COMPATIBILITY, start=34),
        _node("n-sim", NodeKind.SIMULATION, start=35),
        _node("n-gen-b", NodeKind.GENERATED, start=36),
        _node("n-effect-opaque", NodeKind.EFFECT, start=37, confidence=Confidence.OPAQUE),
        _node("n-doc-stale", NodeKind.FILE, start=38, path="docs/stale.md"),
    )
    extra_edges = (
        _edge("e-auth-b-state", EdgeKind.AUTHORIZES, "n-auth-b", "n-state", start=30),
        _edge("e-duplicates", EdgeKind.DUPLICATES, "n-sym-a", "n-sym-b", start=31),
        _edge("e-writes-b1", EdgeKind.WRITES, "n-sym-a", "n-file-b", start=32),
        _edge("e-writes-b2", EdgeKind.WRITES, "n-sym-b", "n-file-b", start=33),
        _edge("e-supersedes-schema", EdgeKind.SUPERSEDES, "n-schema", "n-schema-old", start=34),
        _edge("e-invalidates-receipt", EdgeKind.INVALIDATES, "n-receipt", "n-receipt-old", start=35),
        _edge("e-adapts", EdgeKind.ADAPTS, "n-compat", "n-iface", start=36),
        _edge("e-deprecates", EdgeKind.DEPRECATES, "n-compat", "n-iface", start=37),
        _edge("e-fallback", EdgeKind.FALLBACKS_TO, "n-op", "n-sim", start=38, confidence=Confidence.HEURISTIC),
        _edge("e-generates-b", EdgeKind.GENERATES, "n-gen-b", "n-art", start=39),
        _edge("e-calls-heuristic", EdgeKind.CALLS, "n-op", "n-sym-b", start=40, confidence=Confidence.HEURISTIC),
        _edge("e-mutates-opaque", EdgeKind.MUTATES, "n-effect-opaque", "n-receipt-old", start=41, confidence=Confidence.OPAQUE),
    )
    return _graph(_clean_nodes() + extra_nodes, _clean_edges() + extra_edges)


def _empty_graph() -> ArchitectureIR:
    return _graph((), ())


def _measure(graph: ArchitectureIR) -> SemanticEntropyReport:
    return measure_semantic_entropy(graph, frozen_task_corpus=_CORPUS)


def _numerators(report: SemanticEntropyReport) -> dict[str, int]:
    return {item.kind.value: item.numerator for item in report.dimensions}


def test_closed_dimension_vocabulary_and_evidence_pins() -> None:
    assert SEMANTIC_ENTROPY_SCHEMA == (
        "ipfs_accelerate_py/agent-supervisor/semantic-entropy-report@1"
    )
    assert SEMANTIC_ENTROPY_SCHEMA.endswith("semantic-entropy-report@1")
    assert SEMANTIC_ENTROPY_VERSION == 1
    assert SEMANTIC_ENTROPY_EVIDENCE == "pcar/semantic-entropy-report@1"
    assert ENTROPY_REPORT_EVIDENCE == "pcar/entropy-report@1"
    assert ENTROPY_DIMENSION_SCHEMA.endswith("semantic-entropy-dimension@1")
    assert CHANGE_AMPLIFICATION_SCHEMA.endswith("change-amplification@1")
    assert FROZEN_TASK_CORPUS_SCHEMA.endswith("frozen-task-corpus@1")
    assert ENTROPY_EXTRACTOR_IDENTITY == "pcar-004-semantic-entropy"
    assert RANKING_IS_NON_PROBATIVE is True
    assert ENTROPY_IS_PRIORITIZATION_ONLY is True
    assert tuple(kind.value for kind in REQUIRED_ENTROPY_DIMENSIONS) == (
        "AuthorityMultiplicity",
        "ImplementationDuplication",
        "PublicSurfaceArea",
        "DependencyConeSize",
        "DynamicDispatchUncertainty",
        "StateOwnershipAmbiguity",
        "EffectOpacity",
        "CompatibilityBurden",
        "ValidationAmplification",
        "CacheFragmentation",
        "SchemaDrift",
        "ReceiptDrift",
        "DocumentationDrift",
        "MergeConflictDensity",
        "ContextBurden",
    )
    assert CLOSED_ENTROPY_DIMENSIONS == {kind.value for kind in EntropyDimensionKind}
    assert CLOSED_SAFETY_PREDICATES == set(NON_COMPENSABLE_INVARIANTS)
    with pytest.raises(ValueError):
        EntropyDimensionKind("AestheticScore")


def test_all_dimensions_serialize_independently_with_evidence_and_uncertainty() -> None:
    report = _measure(_clean_graph())
    assert [item.kind for item in report.dimensions] == list(REQUIRED_ENTROPY_DIMENSIONS)
    identities = set()
    for record in report.dimensions:
        payload = record.to_dict()
        restored = EntropyDimensionRecord.from_mapping(payload)
        assert restored == record
        assert payload["schema"] == ENTROPY_DIMENSION_SCHEMA
        assert payload["kind"] == record.kind.value
        assert "numerator" in payload
        assert "denominator" in payload
        assert payload["unit"]
        assert payload["uncertainty"]["confidence"] in {
            "exact",
            "conservative",
            "heuristic",
            "opaque",
        }
        assert payload["evidence"]["extractor_identity"] == ENTROPY_EXTRACTOR_IDENTITY
        assert payload["evidence_identity"] == cid_for_dag_json(payload["evidence"])
        claimed = payload.pop("content_identity")
        validate_cid(claimed, codecs=("dag-json",))
        assert claimed == cid_for_dag_json(payload)
        assert "dimensions" not in restored.to_dict()
        assert "score" not in restored.to_dict()
        identities.add(claimed)
    assert len(identities) == len(REQUIRED_ENTROPY_DIMENSIONS)


def test_report_round_trip_and_canonical_identity() -> None:
    report = _measure(_noisy_graph())
    payload = report.to_dict()
    restored = SemanticEntropyReport.from_mapping(payload)
    assert restored == report
    assert restored.to_dict() == payload
    assert restored.to_json() == json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    assert SemanticEntropyReport.from_json(restored.to_json()) == report
    claimed = payload.pop("content_identity")
    validate_cid(claimed, codecs=("dag-json",))
    assert claimed == cid_for_dag_json(payload)
    assert claimed == report.content_identity
    assert not claimed.startswith("sha256:")
    reversed_dimensions = tuple(reversed(report.dimensions))
    reordered = SemanticEntropyReport(
        architecture_ir_identity=report.architecture_ir_identity,
        repository_tree=report.repository_tree,
        freshness=report.freshness,
        frozen_task_corpus=report.frozen_task_corpus,
        dimensions=reversed_dimensions,
        change_amplification=report.change_amplification,
    )
    assert reordered.content_identity == report.content_identity
    assert [item.kind for item in reordered.dimensions] == list(REQUIRED_ENTROPY_DIMENSIONS)


def test_canonical_vectors_for_clean_and_empty_graphs() -> None:
    empty = _measure(_empty_graph())
    assert canonical_entropy_vector(empty) == tuple(
        (kind.value, 0, None, empty.dimension(kind).unit)
        for kind in REQUIRED_ENTROPY_DIMENSIONS
    )
    for record in empty.dimensions:
        assert record.uncertainty.confidence is Confidence.CONSERVATIVE
        assert record.uncertainty.unknown_denominator == 1
    clean = _measure(_clean_graph())
    vector = {item[0]: item[1:] for item in canonical_entropy_vector(clean)}
    assert vector["AuthorityMultiplicity"] == (0, 2, "authorities")
    assert clean.dimension("AuthorityMultiplicity").numerator == 0
    assert clean.dimension("AuthorityMultiplicity").denominator == 2
    assert clean.dimension("ImplementationDuplication").numerator == 0
    assert clean.dimension("ImplementationDuplication").denominator == 5
    assert clean.dimension("PublicSurfaceArea").numerator == 2
    assert clean.dimension("PublicSurfaceArea").denominator == 5
    assert clean.dimension("DynamicDispatchUncertainty").numerator == 0
    assert clean.dimension("DynamicDispatchUncertainty").denominator == 2
    assert clean.dimension("StateOwnershipAmbiguity").numerator == 0
    assert clean.dimension("StateOwnershipAmbiguity").denominator == 1
    assert clean.dimension("EffectOpacity").numerator == 0
    assert clean.dimension("EffectOpacity").denominator == 3
    assert clean.dimension("CompatibilityBurden").numerator == 0
    assert clean.dimension("CompatibilityBurden").denominator == 2
    assert clean.dimension("ValidationAmplification").numerator == 2
    assert clean.dimension("ValidationAmplification").denominator == 2
    assert clean.dimension("CacheFragmentation").numerator == 0
    assert clean.dimension("CacheFragmentation").denominator == 1
    assert clean.dimension("SchemaDrift").numerator == 0
    assert clean.dimension("SchemaDrift").denominator == 1
    assert clean.dimension("ReceiptDrift").numerator == 0
    assert clean.dimension("ReceiptDrift").denominator == 1
    assert clean.dimension("DocumentationDrift").numerator == 0
    assert clean.dimension("DocumentationDrift").denominator == 1
    assert clean.dimension("MergeConflictDensity").numerator == 0
    assert clean.dimension("MergeConflictDensity").denominator == 4
    assert clean.dimension("DependencyConeSize").numerator > 0
    assert clean.dimension("DependencyConeSize").denominator == 2
    assert clean.dimension("ContextBurden").numerator > 0
    assert clean.dimension("ContextBurden").denominator == 2
    assert clean.dimension("AuthorityMultiplicity").uncertainty.confidence is Confidence.EXACT


def test_noisy_graph_raises_independent_dimension_numerators() -> None:
    clean = _numerators(_measure(_clean_graph()))
    noisy = _numerators(_measure(_noisy_graph()))
    assert noisy["AuthorityMultiplicity"] > clean["AuthorityMultiplicity"]
    assert noisy["ImplementationDuplication"] > clean["ImplementationDuplication"]
    assert noisy["DynamicDispatchUncertainty"] > clean["DynamicDispatchUncertainty"]
    assert noisy["StateOwnershipAmbiguity"] > clean["StateOwnershipAmbiguity"]
    assert noisy["EffectOpacity"] > clean["EffectOpacity"]
    assert noisy["CompatibilityBurden"] > clean["CompatibilityBurden"]
    assert noisy["CacheFragmentation"] > clean["CacheFragmentation"]
    assert noisy["SchemaDrift"] > clean["SchemaDrift"]
    assert noisy["ReceiptDrift"] > clean["ReceiptDrift"]
    assert noisy["DocumentationDrift"] > clean["DocumentationDrift"]
    assert noisy["MergeConflictDensity"] > clean["MergeConflictDensity"]
    report = _measure(_noisy_graph())
    assert report.dimension("AuthorityMultiplicity").numerator == 1
    assert report.dimension("ImplementationDuplication").numerator == 1
    assert report.dimension("StateOwnershipAmbiguity").numerator == 1
    assert report.dimension("SchemaDrift").numerator == 2
    assert report.dimension("ReceiptDrift").numerator == 2
    assert report.dimension("CacheFragmentation").numerator == 1
    assert report.dimension("MergeConflictDensity").numerator == 1
    assert report.dimension("CompatibilityBurden").numerator == 3
    assert report.dimension("DynamicDispatchUncertainty").numerator == 2
    assert report.dimension("EffectOpacity").numerator == 2
    assert report.dimension("DocumentationDrift").numerator == 1
    assert report.dimension("DynamicDispatchUncertainty").uncertainty.confidence is Confidence.HEURISTIC
    assert report.dimension("EffectOpacity").uncertainty.confidence is Confidence.OPAQUE


def test_change_amplification_is_independent_and_documented() -> None:
    report = _measure(_clean_graph())
    measure = report.change_amplification
    assert measure.schema == CHANGE_AMPLIFICATION_SCHEMA
    assert measure.unit == "amplified_units"
    assert measure.frozen_task_corpus_identity == _CORPUS.content_identity
    assert measure.tokens == 0
    assert measure.raw_expansions == 0
    assert measure.uncertainty.unknown_numerator >= 2
    documented = (
        measure.files
        + measure.symbols
        + measure.interfaces
        + measure.schemas
        + measure.effects
        + measure.tests
        + measure.proofs
        + measure.providers
        + measure.runtime_paths
        + measure.owners
        + measure.hops
    )
    assert measure.numerator == documented
    payload = measure.to_dict()
    restored = ChangeAmplificationMeasure.from_mapping(payload)
    assert restored == measure
    assert "score" not in payload
    noisy = _measure(_noisy_graph()).change_amplification
    assert noisy.numerator >= measure.numerator
    assert measure_change_amplification(_clean_graph(), frozen_task_corpus=_CORPUS) == measure


def test_unknown_aggregate_fields_are_rejected() -> None:
    report = _measure(_clean_graph())
    payload = report.to_dict()
    for field in ("aggregate_score", "score", "grade", "composite", "safety", "ranking"):
        forged = dict(payload)
        forged[field] = 0
        identity_payload = {key: value for key, value in forged.items() if key != "content_identity"}
        forged["content_identity"] = cid_for_dag_json(identity_payload)
        with pytest.raises(EntropyContractError, match="undocumented aggregation"):
            SemanticEntropyReport.from_mapping(forged)
    dimension = report.dimensions[0].to_dict()
    dimension["safety_score"] = 0
    identity_payload = {key: value for key, value in dimension.items() if key != "content_identity"}
    dimension["content_identity"] = cid_for_dag_json(identity_payload)
    with pytest.raises(EntropyContractError, match="undocumented aggregation"):
        EntropyDimensionRecord.from_mapping(dimension)
    unknown = dict(payload)
    unknown["hidden"] = True
    identity_payload = {key: value for key, value in unknown.items() if key != "content_identity"}
    unknown["content_identity"] = cid_for_dag_json(identity_payload)
    with pytest.raises(EntropyContractError, match="unknown semantic-entropy field"):
        SemanticEntropyReport.from_mapping(unknown)


def test_missing_fields_and_incomplete_dimension_closure_are_rejected() -> None:
    report = _measure(_clean_graph())
    payload = report.to_dict()
    missing = {key: value for key, value in payload.items() if key != "freshness"}
    with pytest.raises(EntropyContractError, match="missing semantic-entropy field"):
        SemanticEntropyReport.from_mapping(missing)
    truncated = dict(payload)
    truncated["dimensions"] = payload["dimensions"][1:]
    identity_payload = {key: value for key, value in truncated.items() if key != "content_identity"}
    truncated["content_identity"] = cid_for_dag_json(identity_payload)
    with pytest.raises(EntropyContractError, match="missing entropy dimensions"):
        SemanticEntropyReport.from_mapping(truncated)
    duplicated = dict(payload)
    duplicated["dimensions"] = payload["dimensions"] + payload["dimensions"][:1]
    identity_payload = {key: value for key, value in duplicated.items() if key != "content_identity"}
    duplicated["content_identity"] = cid_for_dag_json(identity_payload)
    with pytest.raises(EntropyContractError, match="entropy dimensions must be unique"):
        SemanticEntropyReport.from_mapping(duplicated)


def test_versioned_schema_is_closed() -> None:
    payload = _measure(_clean_graph()).to_dict()
    payload["schema"] = payload["schema"] + "-extra"
    identity_payload = {key: value for key, value in payload.items() if key != "content_identity"}
    payload["content_identity"] = cid_for_dag_json(identity_payload)
    with pytest.raises(EntropyContractError, match="unexpected semantic-entropy schema"):
        SemanticEntropyReport.from_mapping(payload)
    versioned = _measure(_clean_graph()).to_dict()
    versioned["version"] = 2
    identity_payload = {key: value for key, value in versioned.items() if key != "content_identity"}
    versioned["content_identity"] = cid_for_dag_json(identity_payload)
    with pytest.raises(EntropyContractError, match="unexpected semantic-entropy version"):
        SemanticEntropyReport.from_mapping(versioned)


def test_content_identity_mismatch_is_rejected() -> None:
    payload = _measure(_clean_graph()).to_dict()
    payload["content_identity"] = "sha256:" + ("00" * 32)
    with pytest.raises(EntropyContractError, match="content identity mismatch"):
        SemanticEntropyReport.from_mapping(payload)


def test_deterministic_calculation_is_order_independent() -> None:
    graph = _noisy_graph()
    reversed_graph = ArchitectureIR.from_parts(
        repository_tree=_TREE,
        freshness=_FRESHNESS,
        nodes=tuple(reversed(graph.nodes)),
        edges=tuple(reversed(graph.edges)),
    )
    first = _measure(graph)
    second = _measure(reversed_graph)
    assert first.content_identity == second.content_identity
    assert canonical_entropy_vector(first) == canonical_entropy_vector(second)
    assert first.change_amplification == second.change_amplification
    assert measure_entropy_dimensions(graph, frozen_task_corpus=_CORPUS) == first.dimensions


def test_independent_measures_do_not_move_together() -> None:
    baseline = _numerators(_measure(_clean_graph()))
    duplicated = _graph(
        _clean_nodes(),
        _clean_edges()
        + (_edge("e-duplicates", EdgeKind.DUPLICATES, "n-sym-a", "n-sym-b", start=50),),
    )
    after_duplicate = _numerators(_measure(duplicated))
    changed = {
        kind for kind, value in after_duplicate.items() if value != baseline[kind]
    }
    assert changed == {"ImplementationDuplication"}
    assert after_duplicate["ImplementationDuplication"] == baseline["ImplementationDuplication"] + 1

    extra_authority = _graph(
        _clean_nodes() + (_node("n-auth-extra", NodeKind.AUTHORITY, start=51),),
        _clean_edges(),
    )
    after_authority = _numerators(_measure(extra_authority))
    changed = {
        kind for kind, value in after_authority.items() if value != baseline[kind]
    }
    assert changed == {"AuthorityMultiplicity"}

    extra_iface = _graph(
        _clean_nodes() + (_node("n-iface-extra", NodeKind.INTERFACE, start=52),),
        _clean_edges(),
    )
    after_iface = _numerators(_measure(extra_iface))
    changed = {kind for kind, value in after_iface.items() if value != baseline[kind]}
    assert changed == {"PublicSurfaceArea"}

    heuristic = _numerators(_measure(_clean_graph(calls_confidence=Confidence.HEURISTIC)))
    changed = {kind for kind, value in heuristic.items() if value != baseline[kind]}
    assert changed == {"DynamicDispatchUncertainty"}
    assert heuristic["DynamicDispatchUncertainty"] == 1
    heuristic_report = _measure(_clean_graph(calls_confidence=Confidence.HEURISTIC))
    assert heuristic_report.dimension("DynamicDispatchUncertainty").uncertainty.confidence is (
        Confidence.HEURISTIC
    )
    assert heuristic_report.dimension("DependencyConeSize").numerator == (
        _measure(_clean_graph()).dimension("DependencyConeSize").numerator
    )


def test_monotonic_fixture_properties() -> None:
    clean = _measure(_clean_graph())
    with_duplicate = _measure(
        _graph(
            _clean_nodes(),
            _clean_edges()
            + (_edge("e-duplicates", EdgeKind.DUPLICATES, "n-sym-a", "n-sym-b", start=60),),
        )
    )
    assert (
        with_duplicate.dimension("ImplementationDuplication").numerator
        >= clean.dimension("ImplementationDuplication").numerator
    )
    with_conflict = _measure(
        _graph(
            _clean_nodes() + (_node("n-file-b", NodeKind.FILE, start=61),),
            _clean_edges()
            + (
                _edge("e-writes-b1", EdgeKind.WRITES, "n-sym-a", "n-file-b", start=62),
                _edge("e-writes-b2", EdgeKind.WRITES, "n-sym-b", "n-file-b", start=63),
            ),
        )
    )
    assert (
        with_conflict.dimension("MergeConflictDensity").numerator
        >= clean.dimension("MergeConflictDensity").numerator
    )
    noisy = _measure(_noisy_graph())
    for kind in REQUIRED_ENTROPY_DIMENSIONS:
        assert noisy.dimension(kind).numerator >= clean.dimension(kind).numerator


def test_unbound_corpus_widens_context_and_amplification_uncertainty() -> None:
    bound = measure_semantic_entropy(_clean_graph(), frozen_task_corpus=_CORPUS)
    unbound = measure_semantic_entropy(_clean_graph())
    assert unbound.frozen_task_corpus.corpus_id == "pcar-004-unbound-corpus"
    assert unbound.frozen_task_corpus.task_ids == ()
    assert bound.dimension("ContextBurden").uncertainty.unknown_numerator == 0
    assert unbound.dimension("ContextBurden").uncertainty.unknown_numerator == 2
    assert unbound.dimension("ContextBurden").uncertainty.confidence is Confidence.CONSERVATIVE
    assert unbound.change_amplification.uncertainty.unknown_numerator > (
        bound.change_amplification.uncertainty.unknown_numerator
    )
    assert unbound.change_amplification.uncertainty.confidence is Confidence.CONSERVATIVE


def test_lower_scores_cannot_satisfy_safety_predicate() -> None:
    empty = _measure(_empty_graph())
    noisy = _measure(_noisy_graph())
    assert all(item.numerator == 0 for item in empty.dimensions)
    assert empty.change_amplification.numerator == 0
    ranking = derive_non_probative_ranking(empty)
    assert ranking
    assert "ranking" not in empty.to_dict()
    assert RANKING_IS_NON_PROBATIVE is True
    for predicate in NON_COMPENSABLE_INVARIANTS:
        assert entropy_satisfies_safety_predicate(empty, predicate) is False
        assert entropy_satisfies_safety_predicate(noisy, predicate) is False
    for claim in CLOSED_NON_AUTHORITY_CLAIMS:
        assert entropy_establishes(empty, claim) is False
        assert entropy_establishes(noisy, claim) is False
    with pytest.raises(EntropyAuthorityError, match="cannot establish promotion"):
        refuse_entropy_authority("promotion")
    with pytest.raises(EntropyAuthorityError, match="cannot establish deletion"):
        refuse_entropy_authority("deletion")
    with pytest.raises(EntropyAuthorityError, match="cannot establish safety"):
        refuse_entropy_authority("safety")
    with pytest.raises(EntropyContractError, match="unsupported safety predicate"):
        entropy_satisfies_safety_predicate(empty, "LooksSafeBecauseScoreIsZero")


def test_frozen_task_corpus_is_retained_and_versioned() -> None:
    report = _measure(_clean_graph())
    corpus_payload = report.frozen_task_corpus.to_dict()
    assert corpus_payload["schema"] == FROZEN_TASK_CORPUS_SCHEMA
    assert corpus_payload["task_ids"] == ["PCAR-004"]
    restored = FrozenTaskCorpus.from_mapping(corpus_payload)
    assert restored == _CORPUS
    claimed = corpus_payload.pop("content_identity")
    assert claimed == cid_for_dag_json(corpus_payload)
    assert report.change_amplification.frozen_task_corpus_identity == claimed


def test_zero_denominator_is_not_serialized_as_success() -> None:
    record = _measure(_empty_graph()).dimension("ValidationAmplification")
    assert record.numerator == 0
    assert record.denominator is None
    assert record.ratio_basis_points() is None
    assert record.to_dict()["denominator"] is None
    assert record.uncertainty.unknown_denominator == 1
