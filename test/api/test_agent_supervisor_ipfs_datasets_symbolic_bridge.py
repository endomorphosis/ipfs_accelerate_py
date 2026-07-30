"""SCA-213 / SCA-626 real-module canary for exact GraphRAG and Cypher-AST binding.

Proves objective evidence SCAEV031DATASETSGRAPH for SCA-G031.

Exercises ``ipfs_datasets_py.logic.intent_ir.graphrag.retrieval.IntentGraphRetriever``
and ``ipfs_datasets_py.knowledge_graphs.cypher.ast`` / ``parser`` through the
supervisor facades.  Signatures and identities are capability-receipted.
GraphRAG and Cypher AST remain non-authoritative.  Missing or incompatible
modules are typed blockers.  Fixture-only success is rejected.
"""

from __future__ import annotations

import importlib

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.symbolic_contract_graph import (
    BOUNDED_GRAPHRAG_RETRIEVER_INTERFACE,
    EXACT_DATASETS_CYPHER_AST_MODULE,
    EXACT_DATASETS_CYPHER_PARSER_MODULE,
    EXACT_DATASETS_GRAPHRAG_MODULE,
    SCAEV031DATASETSGRAPH as GRAPH_SCAEV031DATASETSGRAPH,
    SCAEV031DATASETSGRAPH_EVIDENCE as GRAPH_SCAEV031DATASETSGRAPH_EVIDENCE,
    BoundedGraphRAGRetriever,
    ContractAuthority,
    ContractEdgeKind,
    ContractGraphEdge,
    ContractGraphNode,
    ContractNodeKind,
    ContractProvenance,
    ExactDatasetsGraphProviderError,
    GRAPH_VERSION,
    RetrievalBounds,
    SymbolicContractGraph,
)
from ipfs_accelerate_py.agent_supervisor.integrations.ipfs_datasets_analysis_provider import (
    DATASETS_GRAPH_BACKEND_SPECS,
    DATASETS_GRAPH_CONTEXT_AUTHORITY,
    DATASETS_GRAPH_SYNTAX_ONLY,
    EXACT_CYPHER_AST_MODULE,
    EXACT_CYPHER_PARSER_MODULE,
    EXACT_GRAPHRAG_RETRIEVAL_MODULE,
    INTENT_GRAPH_RETRIEVER_INTERFACE,
    PACKAGE_ROOT_FALLBACK_MODULES,
    QUERY_NODE_INTERFACE,
    SCAEV031DATASETSGRAPH,
    SCAEV031DATASETSGRAPH_COVERAGE,
    SCAEV031DATASETSGRAPH_EVIDENCE,
    DatasetsGraphBackendError,
    DatasetsGraphBackendKind,
    DatasetsGraphBackendProbe,
    DatasetsGraphSymbolReceipt,
    ExactDatasetsGraphRAGAdapter,
    LocalGraphRAGRetrievalAdapter,
    admit_exact_datasets_graph_source,
    create_exact_datasets_graphrag_adapter,
    inspect_exact_datasets_graph_capability,
    parse_cypher_query_ast,
    probe_all_datasets_graph_backends,
    probe_datasets_graph_backend,
    run_datasets_graph_real_module_canary,
    run_intent_graph_retriever_canary,
)


pytest.importorskip("ipfs_datasets_py")

SNAPSHOT = "repository-snapshot:sha256:sca-213-fixture"


def _node(key: str, *, symbol: str = "", path: str = "") -> ContractGraphNode:
    payload = {"label": key}
    if symbol:
        payload["symbol"] = symbol
    if path:
        payload["path"] = path
    return ContractGraphNode(
        kind=ContractNodeKind.SYMBOL,
        stable_key=key,
        snapshot_id=SNAPSHOT,
        provenance=ContractProvenance.AST,
        authority=ContractAuthority.SOURCE_OBSERVATION,
        version=GRAPH_VERSION,
        payload=payload,
        source_refs=("ast-record:sca-213",),
    )


def _edge(
    source: ContractGraphNode,
    target: ContractGraphNode,
) -> ContractGraphEdge:
    return ContractGraphEdge(
        kind=ContractEdgeKind.DEPENDS_ON,
        source=source.node_id,
        target=target.node_id,
        snapshot_id=SNAPSHOT,
        provenance=ContractProvenance.AST,
        authority=ContractAuthority.SOURCE_OBSERVATION,
        version=GRAPH_VERSION,
        mandatory=True,
        source_refs=("ast-record:sca-213",),
    )


def _graph() -> SymbolicContractGraph:
    alpha = _node("symbol:alpha", symbol="alpha", path="pkg/alpha.py")
    beta = _node("symbol:beta", symbol="beta", path="pkg/beta.py")
    return SymbolicContractGraph(
        snapshot_id=SNAPSHOT,
        nodes=(alpha, beta),
        edges=(_edge(alpha, beta),),
    )


def test_scaev031datasetsgraph_evidence_term_is_declared_and_receipted() -> None:
    """Exact-text SCAEV031DATASETSGRAPH markers for objective evidence admission."""

    assert SCAEV031DATASETSGRAPH == "SCAEV031DATASETSGRAPH"
    assert SCAEV031DATASETSGRAPH_EVIDENCE == SCAEV031DATASETSGRAPH
    assert GRAPH_SCAEV031DATASETSGRAPH == SCAEV031DATASETSGRAPH
    assert GRAPH_SCAEV031DATASETSGRAPH_EVIDENCE == SCAEV031DATASETSGRAPH
    assert "exact-graphrag-intent-graph-retriever-module-binding" in (
        SCAEV031DATASETSGRAPH_COVERAGE
    )
    assert "exact-cypher-ast-and-parser-module-binding" in SCAEV031DATASETSGRAPH_COVERAGE
    assert "real-module-canary-context-only-candidates" in SCAEV031DATASETSGRAPH_COVERAGE
    assert "cypher-ast-syntax-only-non-authoritative" in SCAEV031DATASETSGRAPH_COVERAGE
    assert (
        "package-root-fixture-local-lexical-cannot-claim-exact-use"
        in SCAEV031DATASETSGRAPH_COVERAGE
    )

    capability = inspect_exact_datasets_graph_capability()
    assert capability["evidence_id"] == SCAEV031DATASETSGRAPH
    assert SCAEV031DATASETSGRAPH in capability["evidence"]["requirement_ids"]
    assert capability["evidence"]["coverage"] == list(SCAEV031DATASETSGRAPH_COVERAGE)
    assert capability["evidence"]["evidence_id"] == SCAEV031DATASETSGRAPH_EVIDENCE

    canary = run_datasets_graph_real_module_canary()
    assert canary["evidence_id"] == SCAEV031DATASETSGRAPH
    assert SCAEV031DATASETSGRAPH in canary["evidence"]["requirement_ids"]
    assert canary["evidence"]["coverage"] == list(SCAEV031DATASETSGRAPH_COVERAGE)

    local_capability = BoundedGraphRAGRetriever(_graph()).capability()
    assert local_capability["evidence_id"] == SCAEV031DATASETSGRAPH
    assert SCAEV031DATASETSGRAPH in local_capability["evidence"]["requirement_ids"]


def test_exact_backend_specs_cover_graphrag_and_cypher_ast() -> None:
    kinds = {item for item in DatasetsGraphBackendKind}
    assert kinds == {
        DatasetsGraphBackendKind.GRAPHRAG,
        DatasetsGraphBackendKind.CYPHER_AST,
    }
    assert set(DATASETS_GRAPH_BACKEND_SPECS) == kinds
    assert EXACT_GRAPHRAG_RETRIEVAL_MODULE == EXACT_DATASETS_GRAPHRAG_MODULE
    assert EXACT_CYPHER_AST_MODULE == EXACT_DATASETS_CYPHER_AST_MODULE
    assert EXACT_CYPHER_PARSER_MODULE == EXACT_DATASETS_CYPHER_PARSER_MODULE
    assert INTENT_GRAPH_RETRIEVER_INTERFACE == "IntentGraphRetriever@1"
    assert QUERY_NODE_INTERFACE == "QueryNode@1"
    assert "ipfs_datasets_py" in PACKAGE_ROOT_FALLBACK_MODULES


def test_real_module_probes_bind_intent_graph_retriever_and_cypher_signatures() -> None:
    probes = probe_all_datasets_graph_backends()
    by_kind = {probe.kind: probe for probe in probes}
    assert set(by_kind) == set(DatasetsGraphBackendKind)

    for kind, probe in by_kind.items():
        assert probe.available, (
            f"{kind.value} backend must be available for real-module conformance: "
            f"{probe.unavailable_reason} ({probe.reason_code})"
        )
        assert probe.non_authoritative is True
        assert probe.authoritative is False
        assert probe.symbol_receipts
        assert all(item.available for item in probe.symbol_receipts)
        assert all(item.signature for item in probe.symbol_receipts)
        assert probe.capability_revision.startswith(
            "datasets-graph-capability:sha256:"
        )
        assert probe.package_tree.startswith("datasets-package-tree:sha256:")
        assert all(
            not item.module.startswith("ipfs_datasets_py.")
            or item.module not in PACKAGE_ROOT_FALLBACK_MODULES
            for item in probe.symbol_receipts
        )

    graphrag = by_kind[DatasetsGraphBackendKind.GRAPHRAG]
    assert graphrag.interface == INTENT_GRAPH_RETRIEVER_INTERFACE
    assert EXACT_GRAPHRAG_RETRIEVAL_MODULE in graphrag.module_paths
    assert any(
        item.name == "IntentGraphRetriever" and item.available
        for item in graphrag.symbol_receipts
    )
    retriever_receipt = next(
        item
        for item in graphrag.symbol_receipts
        if item.name == "IntentGraphRetriever"
    )
    assert "retrieve" in retriever_receipt.signature

    cypher = by_kind[DatasetsGraphBackendKind.CYPHER_AST]
    assert cypher.interface == QUERY_NODE_INTERFACE
    assert EXACT_CYPHER_AST_MODULE in cypher.module_paths
    assert EXACT_CYPHER_PARSER_MODULE in cypher.module_paths
    assert any(item.name == "QueryNode" for item in cypher.symbol_receipts)
    assert any(item.name == "CypherParser" for item in cypher.symbol_receipts)
    assert any(item.name == "parse_cypher" for item in cypher.symbol_receipts)


def test_real_module_canary_exercises_intent_graph_retriever_and_cypher_ast() -> None:
    canary = run_datasets_graph_real_module_canary()

    assert canary["exact_module"] is True
    assert canary["fixture_only"] is False
    assert canary["package_root_fallback"] is False
    assert canary["authoritative"] is False
    assert canary["non_authoritative"] is True
    assert canary["proof_authority"] is False
    assert canary["completion_authority"] is False
    assert canary["canary_receipt_id"].startswith("datasets-graph-canary:sha256:")
    assert canary["evidence_id"] == SCAEV031DATASETSGRAPH

    capability = canary["capability"]
    assert capability["available"] is True
    assert capability["package_root_fallback_accepted"] is False
    assert capability["fixture_only_accepted"] is False
    assert capability["local_lexical_fallback_accepted"] is False
    assert capability["exact_modules"]["graphrag"] == EXACT_GRAPHRAG_RETRIEVAL_MODULE
    assert capability["exact_modules"]["cypher_ast"] == EXACT_CYPHER_AST_MODULE
    assert capability["exact_modules"]["cypher_parser"] == EXACT_CYPHER_PARSER_MODULE
    assert capability["capability_receipt_id"]
    assert capability["evidence_id"] == SCAEV031DATASETSGRAPH
    # Real-module canary receipts include version and package-tree identities.
    graphrag_backend = capability["backends"]["graphrag"]
    cypher_backend = capability["backends"]["cypher_ast"]
    assert graphrag_backend["package_version"]
    assert graphrag_backend["package_tree"].startswith("datasets-package-tree:sha256:")
    assert cypher_backend["package_version"]
    assert cypher_backend["package_tree"].startswith("datasets-package-tree:sha256:")

    graphrag = canary["graphrag"]
    assert graphrag["interface"] == INTENT_GRAPH_RETRIEVER_INTERFACE
    assert graphrag["module"] == EXACT_GRAPHRAG_RETRIEVAL_MODULE
    assert graphrag["symbol"] == "IntentGraphRetriever.retrieve"
    assert graphrag["status"] == "ok"
    assert graphrag["authority"] == DATASETS_GRAPH_CONTEXT_AUTHORITY
    assert graphrag["non_authoritative"] is True
    assert graphrag["proof_authority"] is False
    assert graphrag["safe_for_proof"] is False
    assert graphrag["fixture_only"] is False
    assert graphrag["exact_module"] is True
    assert graphrag["graph_root"]
    assert graphrag["graph_digest"]
    assert graphrag["bounds"]
    assert graphrag["candidate_count"] >= 1
    assert graphrag["result_identity"]
    assert graphrag["receipt_id"]
    for premise in graphrag["premises"]:
        assert premise.get("proof_authority") is False
        assert premise.get("authority") in (
            DATASETS_GRAPH_CONTEXT_AUTHORITY,
            "context_only",
        )

    cypher = canary["cypher_ast"]
    assert cypher["interface"] == QUERY_NODE_INTERFACE
    assert cypher["module"] == EXACT_CYPHER_PARSER_MODULE
    assert cypher["ast_module"] == EXACT_CYPHER_AST_MODULE
    assert cypher["ast_class"] == "QueryNode"
    assert cypher["syntax_only"] is True
    assert cypher["authority"] == DATASETS_GRAPH_SYNTAX_ONLY
    assert cypher["source_language_parser"] is False
    assert cypher["non_authoritative"] is True
    assert cypher["proof_authority"] is False
    assert cypher["clause_count"] >= 1
    assert cypher["receipt_id"]


def test_cypher_ast_is_syntax_only_and_non_authoritative() -> None:
    receipt = parse_cypher_query_ast(
        "MATCH (n:Module) WHERE n.name = 'alpha' RETURN n"
    )
    assert receipt["syntax_only"] is True
    assert receipt["authority"] == DATASETS_GRAPH_SYNTAX_ONLY
    assert receipt["authoritative"] is False
    assert receipt["non_authoritative"] is True
    assert receipt["proof_authority"] is False
    assert receipt["completion_authority"] is False
    assert receipt["source_language_parser"] is False
    assert receipt["ast_class"] == "QueryNode"

    # Direct real-module type identity.
    ast_module = importlib.import_module(EXACT_CYPHER_AST_MODULE)
    parser_module = importlib.import_module(EXACT_CYPHER_PARSER_MODULE)
    node = parser_module.CypherParser().parse(
        "MATCH (n:Module) WHERE n.name = 'alpha' RETURN n"
    )
    assert isinstance(node, ast_module.QueryNode)


def test_package_root_fixture_and_local_lexical_cannot_claim_exact_use() -> None:
    probe = probe_datasets_graph_backend(DatasetsGraphBackendKind.GRAPHRAG)
    assert probe.available

    with pytest.raises(DatasetsGraphBackendError, match="cannot claim exact"):
        admit_exact_datasets_graph_source("fixture_only", probe=probe)
    with pytest.raises(DatasetsGraphBackendError, match="cannot claim exact"):
        admit_exact_datasets_graph_source("local_lexical", probe=probe)
    with pytest.raises(DatasetsGraphBackendError, match="cannot claim exact"):
        admit_exact_datasets_graph_source("package_root", probe=probe)
    with pytest.raises(DatasetsGraphBackendError, match="package-root"):
        admit_exact_datasets_graph_source(
            "exact_module",
            probe=probe,
            module_name="ipfs_datasets_py",
        )

    # Capability labels / forged available probes do not satisfy the gate.
    forged = DatasetsGraphBackendProbe(
        kind=DatasetsGraphBackendKind.GRAPHRAG,
        provider_id="intent-graph-retriever",
        available=False,
        interface=INTENT_GRAPH_RETRIEVER_INTERFACE,
        package_version="forged",
        package_tree="",
        capability_revision="forged",
        symbol_receipts=(
            DatasetsGraphSymbolReceipt(
                module="not.a.real.module",
                name="fake",
                qualname="not.a.real.module.fake",
                available=False,
                signature="",
                reason_code="symbol_missing",
            ),
        ),
        reason_code="symbol_missing",
    )
    with pytest.raises(DatasetsGraphBackendError):
        admit_exact_datasets_graph_source("exact_module", probe=forged)

    # Local lexical adapter is not an exact datasets backend.
    local = LocalGraphRAGRetrievalAdapter()
    assert local.operation.value == "graphrag_retrieval"
    with pytest.raises(DatasetsGraphBackendError, match="cannot claim exact"):
        admit_exact_datasets_graph_source("lexical_fallback")


def test_missing_or_incompatible_modules_are_typed_blockers() -> None:
    def missing_importer(name: str) -> object:
        raise ModuleNotFoundError(name)

    probe = probe_datasets_graph_backend(
        DatasetsGraphBackendKind.GRAPHRAG,
        importer=missing_importer,
    )
    assert probe.available is False
    assert probe.reason_code.startswith("module_import_failed")
    assert probe.unavailable_reason

    capability = inspect_exact_datasets_graph_capability(importer=missing_importer)
    assert capability["available"] is False
    assert capability["blockers"]
    assert all(item["reason_code"] for item in capability["blockers"])

    with pytest.raises(DatasetsGraphBackendError) as excinfo:
        run_intent_graph_retriever_canary(importer=missing_importer)
    assert excinfo.value.reason_code.startswith("module_import_failed") or (
        excinfo.value.reason_code == "backend_unavailable"
        or "module_import_failed" in excinfo.value.reason_code
    )

    with pytest.raises(DatasetsGraphBackendError):
        parse_cypher_query_ast("MATCH (n) RETURN n", importer=missing_importer)

    with pytest.raises(DatasetsGraphBackendError):
        create_exact_datasets_graphrag_adapter(importer=missing_importer)

    # Signature-incompatible symbol is also a typed blocker.
    real_import = importlib.import_module

    class EmptyModule:
        pass

    def incompatible_importer(name: str) -> object:
        if name == EXACT_GRAPHRAG_RETRIEVAL_MODULE:
            return EmptyModule()
        return real_import(name)

    incompatible = probe_datasets_graph_backend(
        DatasetsGraphBackendKind.GRAPHRAG,
        importer=incompatible_importer,
    )
    assert incompatible.available is False
    assert incompatible.reason_code == "symbol_missing"


def test_bounded_graphrag_retriever_binds_exact_datasets_without_proof_authority() -> None:
    graph = _graph()
    retriever = BoundedGraphRAGRetriever(graph)

    local_capability = retriever.capability()
    assert local_capability["interface"] == BOUNDED_GRAPHRAG_RETRIEVER_INTERFACE
    assert local_capability["evidence_id"] == SCAEV031DATASETSGRAPH
    assert local_capability["optional_provider_lazy"] is True
    assert local_capability["exact_datasets_lazy"] is True
    assert local_capability["package_root_fallback_accepted"] is False
    assert local_capability["fixture_only_accepted"] is False
    assert local_capability["local_lexical_claims_exact_datasets"] is False
    assert local_capability["non_authoritative"] is True
    assert local_capability["proof_authority"] is False
    assert (
        local_capability["exact_modules"]["graphrag"]
        == EXACT_DATASETS_GRAPHRAG_MODULE
    )
    assert retriever.provider_loaded is False
    assert retriever.exact_datasets_loaded is False

    exact_capability = retriever.exact_datasets_capability()
    assert exact_capability["available"] is True
    assert exact_capability["capability_receipt_id"]
    assert exact_capability["non_authoritative"] is True
    assert exact_capability["proof_authority"] is False

    # Local-only retrieval still works and does not claim exact datasets use.
    local_receipt = retriever.retrieve("alpha beta", bounds=RetrievalBounds(max_candidates=4))
    assert local_receipt.non_authoritative is True
    assert local_receipt.safe_for_proof is False
    assert local_receipt.provider_requested is False
    assert local_receipt.provider_status == "not_requested"
    assert all(
        "ipfs_datasets_exact" not in item.nominated_by
        for item in local_receipt.candidates
    )

    # Exact datasets request capability-receipts IntentGraphRetriever.
    exact_receipt = retriever.retrieve(
        "alpha",
        bounds=RetrievalBounds(max_candidates=4),
        use_exact_datasets=True,
    )
    assert exact_receipt.non_authoritative is True
    assert exact_receipt.safe_for_proof is False
    assert exact_receipt.provider_requested is True
    assert exact_receipt.provider_status == "exact_datasets:completed"
    assert exact_receipt.provider_receipt_id
    assert exact_receipt.reason_code.startswith("exact_datasets_bounded_candidates")
    assert retriever.exact_datasets_loaded is True

    view = retriever.view(
        "alpha",
        retrieval_bounds=RetrievalBounds(max_candidates=4),
        use_exact_datasets=True,
    )
    assert view.retrieval.non_authoritative is True
    assert view.retrieval.safe_for_proof is False
    assert view.mandatory_closure is not None


def test_require_exact_datasets_fails_closed_on_missing_modules() -> None:
    graph = _graph()

    def missing_importer(name: str) -> object:
        raise ModuleNotFoundError(name)

    retriever = BoundedGraphRAGRetriever(
        graph, exact_datasets_importer=missing_importer
    )
    with pytest.raises(ExactDatasetsGraphProviderError) as excinfo:
        retriever.retrieve(
            "alpha",
            use_exact_datasets=True,
            require_exact_datasets=True,
        )
    assert excinfo.value.reason_code in {
        "exact_datasets_unavailable",
        "backend_unavailable",
    } or excinfo.value.reason_code.startswith("module_import_failed")

    # Soft exact request degrades to local without claiming exact success.
    soft = retriever.retrieve("alpha", use_exact_datasets=True)
    assert soft.non_authoritative is True
    assert soft.provider_status.startswith("exact_blocked:")
    assert soft.reason_code == "exact_datasets_blocked_local_fallback"
    assert all(
        "ipfs_datasets_exact" not in item.nominated_by
        for item in soft.candidates
    )


def test_exact_adapter_rejects_fixture_only_and_stays_context_only() -> None:
    adapter = create_exact_datasets_graphrag_adapter()
    capability = adapter.capability()
    assert capability["available"] is True
    assert capability["module"] == EXACT_GRAPHRAG_RETRIEVAL_MODULE
    assert capability["non_authoritative"] is True
    assert capability["proof_authority"] is False
    assert capability["authoritative"] is False

    result = adapter.retrieve_candidates(
        query="canary",
        graph_root="repository:sca-213",
        snapshot_id=SNAPSHOT,
        bounds={"max_results": 1, "max_bytes": 32_000},
    )
    assert result["exact_module"] is True
    assert result["fixture_only"] is False
    assert result["package_root_fallback"] is False
    assert result["non_authoritative"] is True
    assert result["proof_authority"] is False
    assert result["completion_authority"] is False
    assert result["safe_for_proof"] is False
    assert result["authority"] == DATASETS_GRAPH_CONTEXT_AUTHORITY
    assert result["receipt_id"]
    assert result["capability_revision"]
    assert result["package_tree"]
    assert EXACT_GRAPHRAG_RETRIEVAL_MODULE in result["module_paths"]

    # Fixture-only LocalGraphRAG cannot be substituted as ExactDatasetsGraphRAGAdapter.
    assert not isinstance(LocalGraphRAGRetrievalAdapter(), ExactDatasetsGraphRAGAdapter)
