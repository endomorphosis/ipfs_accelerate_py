"""Focused tests for lazy, fail-closed change-propagation capability admission."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

from ipfs_accelerate_py.agent_supervisor.integrations.change_propagation_capabilities import (
    PINNED_TYPESCRIPT_VERSION,
    ChangePropagationCapabilityStatus,
    ChangePropagationDiagnosticCode,
    probe_change_propagation_capabilities,
)


def _module(path: str, **values):
    return SimpleNamespace(__file__=path, **values)


def _logic_provider():
    class Kind:
        def __init__(self, value):
            self.value = value

    class Receipt:
        def __init__(self, module):
            self.module = module
            self.available = True

    class Probe:
        def __init__(self, kind):
            self.kind = Kind(kind)
            self.available = True
            self.symbol_receipts = (Receipt(f"datasets.{kind}"),)
            self.reconstruction_compatible = kind != "ir"
            self.capability_revision = f"revision:{kind}"
            self.package_version = "1.0.0"
            self.provider_id = kind

    class IpfsDatasetsLogicProvider:
        def reconstruct(self, request):  # pragma: no cover - identity only
            return request

    return _module(
        "/fixture/logic_provider.py",
        IpfsDatasetsLogicProvider=IpfsDatasetsLogicProvider,
        DatasetsLogicBackendKind=object(),
        DatasetsLogicBackendProbe=object(),
        LOGIC_IR_INTERFACE="LogicIR@1",
        DATASETS_LOGIC_PROBE_SCHEMA="datasets-probe@1",
        IPFS_DATASETS_LOGIC_PROVIDER_VERSION="1.0.0",
        HAMMER_ADAPTER_SCHEMA_VERSION="hammer@1",
        probe_all_datasets_logic_backends=lambda *, importer: tuple(
            Probe(kind) for kind in ("ir", "tdfol", "cec", "smt", "hammer")
        ),
    )


def _analysis_provider():
    class Kind:
        def __init__(self, value):
            self.value = value

    class Op:
        PREMISE_SELECTION = SimpleNamespace(value="premise_selection")
        GRAPH_RETRIEVAL = SimpleNamespace(value="graph_retrieval")

    class Receipt:
        def __init__(self, module):
            self.module = module
            self.available = True

    class Probe:
        def __init__(self, kind):
            self.kind = Kind(kind)
            self.available = True
            self.symbol_receipts = (Receipt(f"datasets.graph.{kind}"),)
            self.interface = f"{kind}@1"
            self.capability_revision = f"graph-revision:{kind}"
            self.package_version = "1.0.0"
            self.provider_id = kind
            self.unavailable_reason = ""
            self.reason_code = ""

    return _module(
        "/fixture/analysis_provider.py",
        IpfsDatasetsAnalysisProvider=object,
        AnalysisProviderOperation=Op,
        probe_all_datasets_graph_backends=lambda *, importer: tuple(
            Probe(kind) for kind in ("graphrag", "cypher_ast")
        ),
        DATASETS_GRAPH_CAPABILITY_SCHEMA="datasets-graph@1",
        BOUNDED_GRAPHRAG_RETRIEVER_INTERFACE_REF="BoundedGraphRAGRetriever@1",
    )


def _importer(name: str):
    if name == "ipfs_accelerate_py.agent_supervisor.integrations.ipfs_datasets_logic_provider":
        return _logic_provider()
    if name == "ipfs_accelerate_py.agent_supervisor.integrations.ipfs_datasets_analysis_provider":
        return _analysis_provider()
    if name.startswith("datasets."):
        return _module(f"/fixture/{name}.py")
    if name.endswith("repository_indexer"):
        return _module(
            "/fixture/repository_indexer.py",
            RepositoryIndexer=object,
            REPOSITORY_INDEX_SCHEMA="repo-index@1",
            REPOSITORY_INDEXER_VERSION="repository-indexer@1",
        )
    if name.endswith("analysis_ast_index"):
        return _module(
            "/fixture/analysis_ast_index.py",
            AnalysisASTIndex=object,
            build_analysis_ast_index=lambda *a, **k: None,
            ANALYSIS_AST_INDEX_SCHEMA="ast-index@1",
        )
    if name.endswith("code_evidence_graph"):
        return _module(
            "/fixture/code_evidence_graph.py",
            CodeImpactIndex=object,
            CodeEvidenceGraph=object,
            ProvenanceEdge=object,
            EvidenceProvenance=object,
            CODE_IMPACT_INDEX_SCHEMA="impact@1",
            CODE_EVIDENCE_EDGE_SCHEMA="edge@1",
        )
    if name.endswith("semantic_dependency_graph"):
        return _module(
            "/fixture/semantic_dependency_graph.py",
            SemanticDependencyGraph=object,
            build_semantic_dependency_graph=lambda *a, **k: None,
            compute_mandatory_closure=lambda *a, **k: None,
            SEMANTIC_DEPENDENCY_GRAPH_SCHEMA="semantic@1",
        )
    if name.endswith("code_symbol_vector_index"):
        return _module(
            "/fixture/code_symbol_vector_index.py",
            CodeSymbolVectorIndex=object,
            build_code_symbol_vector_index=lambda *a, **k: None,
            search_code_symbol_vector_index=lambda *a, **k: None,
            CODE_SYMBOL_VECTOR_INDEX_SCHEMA="vector@1",
        )
    if name == "ipfs_accelerate_py.llm_router":
        return _module(
            "/fixture/llm_router.py",
            generate_text=lambda *a, **k: "",
            get_last_usage_admission=lambda: {},
            get_last_generation_trace=lambda: {},
        )
    if name.endswith("contract_packet_provider_router"):
        return _module(
            "/fixture/contract_packet_provider_router.py",
            ProviderExecutionReceipt=object,
            PROVIDER_EXECUTION_RECEIPT_INTERFACE="ProviderExecutionReceipt@1",
            PROVIDER_EXECUTION_RECEIPT_SCHEMA="provider-receipt@1",
        )
    # program_graph / program_call_resolver intentionally absent until façades land
    raise ModuleNotFoundError(name)


def _runner(command, **_kwargs):
    executable = Path(command[0]).name
    if executable == "git":
        return SimpleNamespace(
            returncode=0,
            stdout="160000 commit d144be65ffe4c6423e4e1c30cd692812607343eb\tipfs_datasets_py\n",
            stderr="",
        )
    output = {
        "node": "v18.19.1",
        "tsc": "Version 5.5.0",
        "cvc5": "cvc5 version 1.3.3",
        "mypy": "mypy 1.8.0",
        "z3": "Z3 version 4.12.1",
    }.get(executable, "")
    return SimpleNamespace(returncode=0 if output else 1, stdout=output, stderr="")


def _which(executable: str):
    return {
        "node": "/bin/node",
        "tsc": "/bin/tsc",
        "cvc5": "/bin/cvc5",
        "mypy": "/bin/mypy",
        "z3": "/bin/z3",
    }.get(executable)


def test_probe_binds_exact_graph_dataflow_logic_vector_and_llm_surfaces(tmp_path):
    report = probe_change_propagation_capabilities(
        importer=_importer, which=_which, runner=_runner, repository_root=tmp_path
    )

    assert report.datasets_gitlink_revision == "d144be65ffe4c6423e4e1c30cd692812607343eb"

    # Repository / AST indexes
    assert report.capability("index.repository").available
    assert report.capability("index.repository").schema_version == "repo-index@1"
    assert report.capability("index.analysis_ast").available
    assert "ast_index" in report.capability("index.analysis_ast").operations

    # Program graph façades remain typed unavailable until concrete modules exist
    assert (
        report.capability("graph.program_graph").status
        is ChangePropagationCapabilityStatus.UNAVAILABLE
    )
    assert (
        report.capability("graph.program_call_resolver").status
        is ChangePropagationCapabilityStatus.UNAVAILABLE
    )
    assert (
        report.capability("graph.program_graph").diagnostic.code
        is ChangePropagationDiagnosticCode.MODULE_IMPORT_FAILED
    )

    # Impact graph, semantic dependency, value provenance
    assert report.capability("graph.code_impact").available
    assert report.capability("graph.semantic_dependency").available
    assert report.capability("graph.value_provenance").available
    assert "value_provenance" in report.capability("graph.value_provenance").supported_semantics

    # Vector search is non-authoritative nomination only
    vector = report.capability("vector.code_symbol")
    assert vector.available
    assert vector.candidate_authoritative is False
    assert "semantic_authority_false" in vector.supported_semantics
    assert "vector_search" in vector.operations

    # Datasets logic / GraphRAG / premise / reconstruction
    assert report.capability("datasets.logic_ir").available
    assert report.capability("datasets.tdfol").available
    assert report.capability("datasets.cec").available
    assert report.capability("datasets.smt").available
    hammer = report.capability("datasets.hammer")
    assert hammer.available
    assert hammer.candidate_authoritative is False
    assert hammer.reconstruction_compatible is True
    assert report.capability("datasets.reconstruction").available
    assert report.capability("datasets.graphrag").available
    assert report.capability("datasets.cypher_ast").available
    assert report.capability("datasets.premise_selection").available
    assert "premise_selection" in report.capability("datasets.premise_selection").operations

    # Toolchains: python host always available; pinned typescript must match
    assert report.capability("toolchain.python").available
    assert report.capability("toolchain.node").available
    assert report.capability("toolchain.mypy").available
    assert report.capability("toolchain.cvc5").available
    assert report.capability("toolchain.z3").available
    typescript = report.capability("toolchain.typescript")
    assert typescript.status is ChangePropagationCapabilityStatus.INCOMPATIBLE
    assert typescript.details["expected_version"] == PINNED_TYPESCRIPT_VERSION

    # Canonical llm_router and provider receipt APIs
    assert report.capability("llm.router").available
    assert "text.generate" in report.capability("llm.router").operations
    assert "no_completion_authority" in report.capability("llm.router").supported_semantics
    receipt = report.capability("llm.provider_receipt")
    assert receipt.available
    assert receipt.interface_version == "ProviderExecutionReceipt@1"
    assert "no_proof_authority" in receipt.supported_semantics

    payload = report.to_dict()
    assert payload["network_access"] is False
    assert payload["auto_install"] is False
    assert payload["solver_candidates_authoritative"] is False
    assert payload["vector_semantic_authority"] is False
    assert payload["graph_semantic_authority"] is False
    assert payload["llm_completion_authority"] is False


def test_package_presence_does_not_grant_authority_for_missing_symbols(tmp_path):
    def importer(name: str):
        if name.endswith("code_symbol_vector_index"):
            # Package/module present but required search API missing.
            return _module("/fixture/partial_vector.py", CodeSymbolVectorIndex=object)
        return _importer(name)

    report = probe_change_propagation_capabilities(
        importer=importer, which=_which, runner=_runner, repository_root=tmp_path
    )
    vector = report.capability("vector.code_symbol")
    assert vector.status is ChangePropagationCapabilityStatus.PARTIAL
    assert vector.candidate_authoritative is False
    assert vector.diagnostic.code is ChangePropagationDiagnosticCode.REQUIRED_SYMBOL_MISSING


def test_missing_symbol_and_timeout_are_typed_diagnostics(tmp_path):
    def slow_importer(name: str):
        if name.endswith("program_graph"):
            time.sleep(0.05)
        return _importer(name)

    report = probe_change_propagation_capabilities(
        importer=slow_importer,
        which=_which,
        runner=_runner,
        timeout_seconds=0.001,
        repository_root=tmp_path,
    )
    graph = report.capability("graph.program_graph")
    assert graph.status is ChangePropagationCapabilityStatus.TIMED_OUT
    assert graph.diagnostic.code is ChangePropagationDiagnosticCode.PROBE_TIMED_OUT


def test_version_command_timeout_is_typed(tmp_path):
    def runner(command, **_kwargs):
        if Path(command[0]).name == "node":
            raise subprocess.TimeoutExpired(command, 1)
        return _runner(command)

    report = probe_change_propagation_capabilities(
        importer=_importer, which=_which, runner=runner, repository_root=tmp_path
    )
    node = report.capability("toolchain.node")
    assert node.status is ChangePropagationCapabilityStatus.TIMED_OUT
    assert node.diagnostic.code is ChangePropagationDiagnosticCode.PROBE_TIMED_OUT


def test_missing_solver_is_unavailable_not_authoritative(tmp_path):
    def which(executable: str):
        if executable in {"cvc5", "z3"}:
            return None
        return _which(executable)

    report = probe_change_propagation_capabilities(
        importer=_importer, which=which, runner=_runner, repository_root=tmp_path
    )
    cvc5 = report.capability("toolchain.cvc5")
    z3 = report.capability("toolchain.z3")
    assert cvc5.status is ChangePropagationCapabilityStatus.UNAVAILABLE
    assert z3.status is ChangePropagationCapabilityStatus.UNAVAILABLE
    assert cvc5.candidate_authoritative is False
    assert z3.candidate_authoritative is False
    assert cvc5.diagnostic.code is ChangePropagationDiagnosticCode.EXECUTABLE_NOT_FOUND


def test_logic_backend_absence_does_not_admit_reconstruction(tmp_path):
    def importer(name: str):
        if name == "ipfs_accelerate_py.agent_supervisor.integrations.ipfs_datasets_logic_provider":
            class Kind:
                def __init__(self, value):
                    self.value = value

            class Receipt:
                def __init__(self, module):
                    self.module = module
                    self.available = False

            class Probe:
                def __init__(self, kind):
                    self.kind = Kind(kind)
                    self.available = False
                    self.symbol_receipts = (Receipt(f"datasets.{kind}"),)
                    self.reconstruction_compatible = kind != "ir"
                    self.capability_revision = ""
                    self.package_version = ""
                    self.provider_id = kind
                    self.unavailable_reason = "symbols missing"
                    self.reason_code = "backend_unavailable"

            class IpfsDatasetsLogicProvider:
                def reconstruct(self, request):  # pragma: no cover
                    return request

            return _module(
                "/fixture/logic_provider.py",
                IpfsDatasetsLogicProvider=IpfsDatasetsLogicProvider,
                DatasetsLogicBackendKind=object(),
                DatasetsLogicBackendProbe=object(),
                LOGIC_IR_INTERFACE="LogicIR@1",
                DATASETS_LOGIC_PROBE_SCHEMA="datasets-probe@1",
                IPFS_DATASETS_LOGIC_PROVIDER_VERSION="1.0.0",
                HAMMER_ADAPTER_SCHEMA_VERSION="hammer@1",
                probe_all_datasets_logic_backends=lambda *, importer: tuple(
                    Probe(kind) for kind in ("ir", "tdfol", "cec", "smt", "hammer")
                ),
            )
        return _importer(name)

    report = probe_change_propagation_capabilities(
        importer=importer, which=_which, runner=_runner, repository_root=tmp_path
    )
    assert report.capability("datasets.hammer").status is ChangePropagationCapabilityStatus.UNAVAILABLE
    assert (
        report.capability("datasets.reconstruction").status
        is ChangePropagationCapabilityStatus.UNAVAILABLE
    )


def test_report_exposes_unique_capabilities_and_producer_identities(tmp_path):
    report = probe_change_propagation_capabilities(
        importer=_importer, which=_which, runner=_runner, repository_root=tmp_path
    )
    ids = [item.capability_id for item in report.capabilities]
    assert len(ids) == len(set(ids))
    repo = report.capability("index.repository")
    assert repo.producer_id
    assert repo.module_path.startswith("/")
    assert report.operations["llm.router"]
    assert report.interface_versions["llm.provider_receipt"] == "ProviderExecutionReceipt@1"
