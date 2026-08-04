"""Focused tests for lazy, fail-closed Tactician-Hammer capability admission."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

from ipfs_accelerate_py.agent_supervisor.integrations.tactician_hammer_capabilities import (
    IMPORT_ISOLATION_UNSAFE,
    PINNED_CVC5_VERSION,
    ResourceEnforcementStrength,
    TacticianHammerCapabilityStatus,
    TacticianHammerDiagnosticCode,
    probe_tactician_hammer_capabilities,
)


def _module(path: str, **values):
    return SimpleNamespace(__file__=path, **values)


def _unsafe_load_hammer():
    """Stand-in that mirrors the process-global HOME/sys.prefix mutation."""

    import os
    import sys

    original_prefix = sys.prefix
    original_home = os.environ.get("HOME")
    os.environ["HOME"] = "/tmp/fake-symai"
    sys.prefix = "/tmp/fake-symai"
    try:
        return SimpleNamespace()
    finally:
        sys.prefix = original_prefix
        if original_home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = original_home


def _logic_provider():
    return _module(
        "/fixture/logic_provider.py",
        IpfsDatasetsLogicProvider=object,
        DatasetsLogicBackendProbe=object,
        probe_all_datasets_logic_backends=lambda *, importer: (),
        IPFS_DATASETS_LOGIC_PROVIDER_VERSION="1.0.0",
        HAMMER_ADAPTER_SCHEMA_VERSION="hammer@1",
        _load_hammer=_unsafe_load_hammer,
    )


def _analysis_provider():
    class Op:
        PREMISE_SELECTION = SimpleNamespace(value="premise_selection")
        GRAPH_RETRIEVAL = SimpleNamespace(value="graph_retrieval")

    return _module(
        "/fixture/analysis_provider.py",
        IpfsDatasetsAnalysisProvider=object,
        AnalysisProviderOperation=Op,
        probe_all_datasets_graph_backends=lambda *, importer: (),
        DATASETS_GRAPH_CAPABILITY_SCHEMA="datasets-graph@1",
    )


def _importer(name: str):
    # Generic Tactician is intentionally absent (pending LPR-003).
    if name == "ipfs_datasets_py.logic.tactician":
        raise ModuleNotFoundError(name)

    if name == (
        "ipfs_datasets_py.processors.legal_data.proof_tactician"
    ):
        return _module(
            "/fixture/proof_tactician.py",
            ProofTactician=object,
        )

    if name.endswith("ipfs_datasets_logic_provider"):
        return _logic_provider()
    if name.endswith("ipfs_datasets_analysis_provider"):
        return _analysis_provider()

    # Hammer descriptor surfaces.
    if name.endswith("hammers.corpus"):
        return _module(
            "/fixture/hammers_corpus.py",
            CorpusManifest=object,
            TheoremEntry=object,
            CorpusSource=object,
        )
    if name.endswith("hammers.premise_selection"):
        return _module(
            "/fixture/hammers_premise_selection.py",
            select_premises=lambda *a, **k: None,
            GoalFeatures=object,
            PremiseSelectionResult=object,
        )
    if name.endswith("hammers.learned_selector"):
        return _module(
            "/fixture/hammers_learned_selector.py",
            select_premises_gated=lambda *a, **k: None,
            LearnedModelArtifact=object,
            LearnedSelectorConfig=object,
            SelectorFallbackReason=object,
        )
    if name.endswith("hammers.models"):
        return _module(
            "/fixture/hammers_models.py",
            TranslationTarget=object,
            TranslationRecord=object,
            TranslationStatus=object,
            EnvironmentLockRecord=object,
            SCHEMA_VERSION="1.0.0",
        )
    if name.endswith("hammers.translation"):
        return _module(
            "/fixture/hammers_translation.py",
            TranslationMap=object,
            TranslationMapEntry=object,
            TranslationContext=object,
        )
    if name.endswith("hammers.portfolio"):
        return _module(
            "/fixture/hammers_portfolio.py",
            SolverPortfolio=object,
            PortfolioRunResult=object,
            SolverAttemptEvidence=object,
        )
    if name.endswith("hammers.receipts"):
        return _module(
            "/fixture/hammers_receipts.py",
            HammerReceipt=object,
            ReceiptStore=object,
            compute_receipt_digest=lambda *a, **k: "",
        )
    if name.endswith("reconstructors.lean"):
        return _module("/fixture/recon_lean.py", LeanReconstructor=object)
    if name.endswith("reconstructors.coq"):
        return _module("/fixture/recon_coq.py", CoqReconstructor=object)
    if name.endswith("reconstructors.isabelle"):
        return _module("/fixture/recon_isabelle.py", IsabelleReconstructor=object)
    if name.endswith("hammers.reconstruction"):
        return _module(
            "/fixture/hammers_reconstruction.py",
            reconstruct_candidate=lambda *a, **k: None,
            get_reconstructor=lambda *a, **k: None,
            build_environment_lock=lambda *a, **k: None,
        )

    # Static analysis / vector / llm.
    if name.endswith("analysis_ast_index"):
        return _module(
            "/fixture/analysis_ast_index.py",
            AnalysisASTIndex=object,
            build_analysis_ast_index=lambda *a, **k: None,
            ANALYSIS_AST_INDEX_SCHEMA="ast-index@1",
        )
    if name.endswith("program_call_resolver"):
        return _module(
            "/fixture/program_call_resolver.py",
            ProgramCallResolver=object,
            PROGRAM_CALL_RESOLVER_VERSION="program-call-resolver@1",
            PROGRAM_CALL_RESOLVER_SCHEMA="call-resolver@1",
        )
    if name.endswith("code_evidence_graph"):
        return _module(
            "/fixture/code_evidence_graph.py",
            CodeEvidenceGraph=object,
            ProvenanceEdge=object,
            CodeImpactIndex=object,
            CODE_EVIDENCE_GRAPH_SCHEMA="evidence-graph@1",
        )
    if name.endswith("program_contracts"):
        return _module(
            "/fixture/program_contracts.py",
            ExpectedProgramContract=object,
            ObservedProgramContract=object,
            ProgramContractBundle=object,
            PROGRAM_CONTRACT_VERSION=1,
            SCHEMA_VERSION=1,
        )
    if name.endswith("memory_safety_facets"):
        return _module(
            "/fixture/memory_safety_facets.py",
            MemorySafetyEvidenceCollector=object,
            MemorySafetyPolicy=object,
            NativeBoundary=object,
            MEMORY_SAFETY_EVIDENCE_SCHEMA="memory-safety@1",
        )
    if name.endswith("program_graph"):
        return _module(
            "/fixture/program_graph.py",
            ProgramGraph=object,
            ProgramGraphSnapshot=object,
            PROGRAM_GRAPH_VERSION="program-graph@1",
            PROGRAM_GRAPH_SCHEMA="program-graph-schema@1",
        )
    if name.endswith("code_symbol_vector_index"):
        return _module(
            "/fixture/code_symbol_vector_index.py",
            CodeSymbolVectorIndex=object,
            build_code_symbol_vector_index=lambda *a, **k: None,
            search_code_symbol_vector_index=lambda *a, **k: None,
            CODE_SYMBOL_VECTOR_INDEX_SCHEMA="vector@1",
        )
    if name.endswith("change_value_vector_index"):
        return _module(
            "/fixture/change_value_vector_index.py",
            ChangeValueVectorIndex=object,
            CHANGE_VALUE_VECTOR_INDEX_SCHEMA="change-value@1",
        )
    if name == "ipfs_accelerate_py.llm_router":
        return _module(
            "/fixture/llm_router.py",
            generate_text=lambda *a, **k: "",
            get_last_usage_admission=lambda: {},
            get_last_generation_trace=lambda: {},
        )
    raise ModuleNotFoundError(name)


def _runner(command, **_kwargs):
    executable = Path(command[0]).name
    if executable == "git":
        return SimpleNamespace(
            returncode=0,
            stdout=(
                "160000 commit d144be65ffe4c6423e4e1c30cd692812607343eb"
                "\tipfs_datasets_py\n"
            ),
            stderr="",
        )
    output = {
        "z3": "Z3 version 4.12.1",
        "cvc5": "cvc5 version 1.3.3",
        "vampire": "Vampire 4.7",
        "eprover": "E 2.6",
        "lean": "Lean (version 4.31.0)",
        "coqc": "The Rocq Prover, version 9.1.1",
        "isabelle": "Isabelle2024",
        "mypy": "mypy 1.8.0",
    }.get(executable, "")
    # isabelle version is invoked as `isabelle version` — treat argv[0].
    if not output and len(command) > 1 and Path(command[0]).name == "isabelle":
        output = "Isabelle2024"
    return SimpleNamespace(returncode=0 if output else 1, stdout=output, stderr="")


def _which(executable: str):
    return {
        "z3": "/bin/z3",
        "cvc5": "/bin/cvc5",
        "vampire": "/bin/vampire",
        "eprover": "/bin/eprover",
        "lean": "/bin/lean",
        "coqc": "/bin/coqc",
        "isabelle": "/bin/isabelle",
        "mypy": "/bin/mypy",
    }.get(executable)


def test_probe_covers_tactician_hammer_static_vector_kg_and_llm(tmp_path):
    report = probe_tactician_hammer_capabilities(
        importer=_importer, which=_which, runner=_runner, repository_root=tmp_path
    )

    assert report.datasets_gitlink_revision == (
        "d144be65ffe4c6423e4e1c30cd692812607343eb"
    )

    # Domain-neutral Tactician is typed unavailable pending LPR-003.
    generic = report.capability("tactician.generic")
    assert generic.status is TacticianHammerCapabilityStatus.UNAVAILABLE
    assert generic.diagnostic.code is TacticianHammerDiagnosticCode.PENDING_LPR_003
    assert generic.details["pending_task"] == "LPR-003"
    assert generic.details["domain_neutral"] is True

    # Legal ProofTactician retained as legal-adapter-only, not code authority.
    legal = report.capability("tactician.legal_adapter")
    assert legal.available
    assert legal.details["disposition"] == "legal_adapter_only"
    assert legal.details["code_authority"] is False
    assert "legal_adapter_only" in legal.supported_semantics
    assert "not_code_authority" in legal.supported_semantics

    # Hammer corpus, deterministic/learned selectors, translation, portfolio,
    # reconstruction, receipts.
    assert report.capability("hammer.corpus").available
    assert report.capability("hammer.selector.deterministic").available
    learned = report.capability("hammer.selector.learned")
    assert learned.available
    assert learned.details.get("feature_admitted") is False
    assert "opt_in_only" in learned.supported_semantics
    assert report.capability("hammer.translation").available
    assert report.capability("hammer.translation.map").available
    assert report.capability("hammer.portfolio").available
    assert report.capability("hammer.receipt").available
    assert report.capability("hammer.environment_lock").available
    assert (
        "not_signed_binary_integrity"
        in report.capability("hammer.environment_lock").supported_semantics
    )
    assert report.capability("hammer.reconstruction.lean").available
    assert report.capability("hammer.reconstruction.coq").available
    assert report.capability("hammer.reconstruction.isabelle").available
    assert report.capability("hammer.reconstruction.api").available
    assert report.capability("hammer.reconstruction.lean").reconstruction_compatible
    assert report.capability("hammer.portfolio").candidate_authoritative is False

    # Solvers Z3/CVC5/Vampire/E and ITP executables.
    assert report.capability("toolchain.z3").available
    cvc5 = report.capability("toolchain.cvc5")
    assert cvc5.available
    assert cvc5.interface_version == PINNED_CVC5_VERSION
    assert report.capability("toolchain.vampire").available
    assert report.capability("toolchain.e").available
    assert report.capability("toolchain.lean").available
    assert report.capability("toolchain.coq").available
    assert report.capability("toolchain.isabelle").available
    for solver_id in (
        "toolchain.z3",
        "toolchain.cvc5",
        "toolchain.vampire",
        "toolchain.e",
    ):
        assert report.capability(solver_id).candidate_authoritative is False
        assert report.capability(solver_id).details["signed_binary_integrity"] is False

    # AST / call / dataflow / type / effect analyzers.
    assert report.capability("analyzer.ast").available
    assert report.capability("analyzer.call").available
    assert report.capability("analyzer.dataflow").available
    assert report.capability("analyzer.type").available
    assert report.capability("analyzer.effect").available
    assert report.capability("analyzer.program_graph").available

    # Vector / KG / llm_router — non-authoritative.
    vector = report.capability("vector.code_symbol")
    assert vector.available
    assert vector.candidate_authoritative is False
    assert "semantic_authority_false" in vector.supported_semantics
    kg = report.capability("kg.graphrag")
    assert kg.available
    assert "graph_non_authoritative" in kg.supported_semantics
    llm = report.capability("llm.router")
    assert llm.available
    assert "no_completion_authority" in llm.supported_semantics

    # Import isolation unsafe until LPR-012.
    isolation = report.capability("hammer.import_isolation")
    assert isolation.status is TacticianHammerCapabilityStatus.PARTIAL
    assert (
        isolation.diagnostic.code
        is TacticianHammerDiagnosticCode.IMPORT_ISOLATION_UNSAFE
    )
    assert report.import_isolation == IMPORT_ISOLATION_UNSAFE
    assert isolation.details["pending_task"] == "LPR-012"
    assert isolation.details["mutates_home"] is True
    assert isolation.details["mutates_sys_prefix"] is True

    # Platform resource-enforcement strength is typed.
    assert report.resource_enforcement is not None
    assert isinstance(
        report.resource_enforcement.cpu_enforcement, ResourceEnforcementStrength
    )
    assert isinstance(
        report.resource_enforcement.memory_enforcement, ResourceEnforcementStrength
    )
    # Policy network denial is not OS isolation; locks are not signed integrity.
    assert report.resource_enforcement.network_policy_denied is True
    assert report.resource_enforcement.network_os_isolation is False
    assert report.resource_enforcement.signed_binary_integrity is False
    assert report.resource_enforcement.environment_lock_path_version_only is True

    payload = report.to_dict()
    assert payload["network_access"] is False
    assert payload["auto_install"] is False
    assert payload["learned_selector_admitted"] is False
    assert payload["model_execution_admitted"] is False
    assert payload["native_execution_admitted"] is False
    assert payload["network_access_admitted"] is False
    assert payload["auto_install_admitted"] is False
    assert payload["solver_candidates_authoritative"] is False
    assert payload["vector_semantic_authority"] is False
    assert payload["graph_semantic_authority"] is False
    assert payload["llm_completion_authority"] is False
    assert payload["tactician_proof_authority"] is False
    assert payload["network_policy_denied_is_os_isolation"] is False
    assert payload["environment_lock_is_signed_binary_integrity"] is False
    assert payload["legal_tactician_disposition"] == "legal_adapter_only"
    assert payload["import_isolation"] == IMPORT_ISOLATION_UNSAFE

    # Fail-closed feature admissions.
    assert (
        report.capability("feature.native_execution").status
        is TacticianHammerCapabilityStatus.UNAVAILABLE
    )
    assert (
        report.capability("feature.network").diagnostic.code
        is TacticianHammerDiagnosticCode.FEATURE_NOT_ADMITTED
    )
    assert report.capability("feature.network").details["network_os_isolation"] is False
    assert (
        report.capability("feature.auto_install").status
        is TacticianHammerCapabilityStatus.UNAVAILABLE
    )
    assert (
        report.capability("feature.model_execution").status
        is TacticianHammerCapabilityStatus.UNAVAILABLE
    )


def test_package_presence_alone_is_insufficient(tmp_path):
    def importer(name: str):
        if name.endswith("hammers.corpus"):
            # Module present but required symbols missing.
            return _module("/fixture/partial_corpus.py")
        if name.endswith("code_symbol_vector_index"):
            return _module("/fixture/partial_vector.py", CodeSymbolVectorIndex=object)
        return _importer(name)

    report = probe_tactician_hammer_capabilities(
        importer=importer, which=_which, runner=_runner, repository_root=tmp_path
    )
    corpus = report.capability("hammer.corpus")
    assert corpus.status is TacticianHammerCapabilityStatus.PARTIAL
    assert corpus.diagnostic.code is TacticianHammerDiagnosticCode.REQUIRED_SYMBOL_MISSING
    assert corpus.details["package_present"] is True
    assert corpus.candidate_authoritative is False

    vector = report.capability("vector.code_symbol")
    assert vector.status is TacticianHammerCapabilityStatus.PARTIAL
    assert vector.diagnostic.code is TacticianHammerDiagnosticCode.REQUIRED_SYMBOL_MISSING


def test_missing_partial_incompatible_timeout_are_typed(tmp_path):
    def slow_importer(name: str):
        if name.endswith("analysis_ast_index"):
            time.sleep(0.05)
        return _importer(name)

    timed = probe_tactician_hammer_capabilities(
        importer=slow_importer,
        which=_which,
        runner=_runner,
        timeout_seconds=0.001,
        repository_root=tmp_path,
    )
    ast = timed.capability("analyzer.ast")
    assert ast.status is TacticianHammerCapabilityStatus.TIMED_OUT
    assert ast.diagnostic.code is TacticianHammerDiagnosticCode.PROBE_TIMED_OUT

    def importer(name: str):
        if name.endswith("program_contracts"):
            return _module(
                "/fixture/bad_contracts.py",
                ExpectedProgramContract=object,
                ObservedProgramContract=object,
                ProgramContractBundle=object,
                PROGRAM_CONTRACT_VERSION=99,
                SCHEMA_VERSION=1,
            )
        return _importer(name)

    report = probe_tactician_hammer_capabilities(
        importer=importer, which=_which, runner=_runner, repository_root=tmp_path
    )
    type_cap = report.capability("analyzer.type")
    assert type_cap.status is TacticianHammerCapabilityStatus.INCOMPATIBLE
    assert (
        type_cap.diagnostic.code
        is TacticianHammerDiagnosticCode.INTERFACE_VERSION_INCOMPATIBLE
    )

    def which(executable: str):
        if executable == "vampire":
            return None
        return _which(executable)

    missing = probe_tactician_hammer_capabilities(
        importer=_importer, which=which, runner=_runner, repository_root=tmp_path
    )
    vampire = missing.capability("toolchain.vampire")
    assert vampire.status is TacticianHammerCapabilityStatus.UNAVAILABLE
    assert vampire.diagnostic.code is TacticianHammerDiagnosticCode.EXECUTABLE_NOT_FOUND


def test_version_command_timeout_and_incompatible_cvc5(tmp_path):
    def runner(command, **_kwargs):
        name = Path(command[0]).name
        if name == "z3":
            raise subprocess.TimeoutExpired(command, 1)
        if name == "cvc5":
            return SimpleNamespace(
                returncode=0, stdout="cvc5 version 1.0.0", stderr=""
            )
        return _runner(command)

    report = probe_tactician_hammer_capabilities(
        importer=_importer, which=_which, runner=runner, repository_root=tmp_path
    )
    z3 = report.capability("toolchain.z3")
    assert z3.status is TacticianHammerCapabilityStatus.TIMED_OUT
    assert z3.diagnostic.code is TacticianHammerDiagnosticCode.PROBE_TIMED_OUT

    cvc5 = report.capability("toolchain.cvc5")
    assert cvc5.status is TacticianHammerCapabilityStatus.INCOMPATIBLE
    assert cvc5.details["expected_version"] == PINNED_CVC5_VERSION
    assert (
        cvc5.diagnostic.code
        is TacticianHammerDiagnosticCode.EXECUTABLE_VERSION_INCOMPATIBLE
    )


def test_generic_tactician_available_when_exact_descriptors_present(tmp_path):
    def importer(name: str):
        if name == "ipfs_datasets_py.logic.tactician":
            return _module(
                "/fixture/logic_tactician.py",
                LogicTactician=object,
                TacticianPlan=object,
                TacticianPolicy=object,
                TacticianReceipt=object,
                LOGIC_TACTICIAN_INTERFACE="LogicTactician@1",
                TACTICIAN_SCHEMA_VERSION="tactician@1",
            )
        return _importer(name)

    report = probe_tactician_hammer_capabilities(
        importer=importer, which=_which, runner=_runner, repository_root=tmp_path
    )
    generic = report.capability("tactician.generic")
    assert generic.available
    assert "domain_neutral" in generic.supported_semantics
    assert "advisory_only" in generic.supported_semantics
    assert "no_proof_authority" in generic.supported_semantics


def test_explicit_feature_admission_flags(tmp_path):
    report = probe_tactician_hammer_capabilities(
        importer=_importer,
        which=_which,
        runner=_runner,
        repository_root=tmp_path,
        learned_selector_admitted=True,
        model_execution_admitted=True,
        native_execution_admitted=True,
        network_access_admitted=True,
        auto_install_admitted=True,
    )
    assert report.learned_selector_admitted is True
    assert report.model_execution_admitted is True
    assert report.native_execution_admitted is True
    assert report.network_access_admitted is True
    assert report.auto_install_admitted is True
    assert report.capability("feature.native_execution").available
    assert report.capability("feature.network").available
    # Even when network is admitted, it is not claimed as OS isolation.
    assert report.capability("feature.network").details["network_os_isolation"] is False
    assert report.to_dict()["network_policy_denied_is_os_isolation"] is False
    assert report.capability("feature.auto_install").available
    assert report.capability("feature.model_execution").available


def test_probe_does_not_call_unsafe_load_hammer(tmp_path):
    calls: list[str] = []

    def tracking_load_hammer():
        calls.append("loaded")
        return _unsafe_load_hammer()

    def importer(name: str):
        if name.endswith("ipfs_datasets_logic_provider"):
            mod = _logic_provider()
            return _module(
                mod.__file__,
                IpfsDatasetsLogicProvider=object,
                DatasetsLogicBackendProbe=object,
                probe_all_datasets_logic_backends=lambda *, importer: (),
                IPFS_DATASETS_LOGIC_PROVIDER_VERSION="1.0.0",
                HAMMER_ADAPTER_SCHEMA_VERSION="hammer@1",
                _load_hammer=tracking_load_hammer,
            )
        return _importer(name)

    report = probe_tactician_hammer_capabilities(
        importer=importer, which=_which, runner=_runner, repository_root=tmp_path
    )
    assert calls == []
    assert report.import_isolation == IMPORT_ISOLATION_UNSAFE
    # Source inspection still detects the unsafe mutation pattern.
    isolation = report.capability("hammer.import_isolation")
    assert isolation.details["mutates_home"] is True
    assert isolation.details["mutates_sys_prefix"] is True
