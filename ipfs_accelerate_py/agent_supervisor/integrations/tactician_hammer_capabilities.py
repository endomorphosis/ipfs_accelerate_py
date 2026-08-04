"""Lazy, fail-closed capability admission for Tactician-Hammer logic repair.

Importing this module is deliberately cheap: optional datasets Tactician /
Hammer surfaces, static-analysis facades, vector/KG providers, solver
executables, reconstruction toolchains, and ``llm_router`` are inspected only
by :func:`probe_tactician_hammer_capabilities`.

Discovery is not authority.  Package presence, solver binaries, vector hits,
knowledge-graph edges, learned ranking, and model proposals never promote a
capability to semantic, proof, or completion authority.  Missing, partial,
incompatible, or timed-out surfaces yield typed diagnostics.

The probe never installs packages, invokes package managers, contacts a
network service, executes proof search, or calls the production Hammer
lazy-load path that mutates process-global ``HOME`` / ``sys.prefix``.  That
load path is reported as ``import_isolation_unsafe`` until LPR-012 hardens it.

Policy-declared network denial is not OS isolation.  Path/version environment
locks are not signed binary integrity.  Learned, model, native-execution, and
install features remain off unless an explicit admission flag is supplied.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import os
import platform
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from .contract_repair_dependencies import (
    PINNED_CVC5_VERSION,
    find_contract_repair_executable,
)

TACTICIAN_HAMMER_CAPABILITY_SCHEMA_VERSION: Final = (
    "ipfs_accelerate_py/agent-supervisor/tactician-hammer-capability@1"
)
TACTICIAN_HAMMER_CAPABILITY_REPORT_SCHEMA_VERSION: Final = (
    "ipfs_accelerate_py/agent-supervisor/tactician-hammer-capability-report@1"
)
TACTICIAN_HAMMER_CAPABILITY_REPORT_VERSION: Final = 1
DEFAULT_CAPABILITY_PROBE_TIMEOUT_SECONDS: Final = 10.0

# Explicit until LPR-012 replaces process-global HOME/sys.prefix mutation.
IMPORT_ISOLATION_UNSAFE: Final = "import_isolation_unsafe"
IMPORT_ISOLATION_HARDENED: Final = "import_isolation_hardened"

# Domain-neutral Tactician expected after LPR-003.
GENERIC_TACTICIAN_MODULE: Final = "ipfs_datasets_py.logic.tactician"
GENERIC_TACTICIAN_SYMBOLS: Final = (
    "LogicTactician",
    "TacticianPlan",
    "TacticianPolicy",
    "TacticianReceipt",
)
LEGAL_TACTICIAN_MODULE: Final = (
    "ipfs_datasets_py.processors.legal_data.proof_tactician"
)
LEGAL_TACTICIAN_SYMBOLS: Final = ("ProofTactician",)
LOGIC_PROVIDER_MODULE: Final = (
    "ipfs_accelerate_py.agent_supervisor.integrations.ipfs_datasets_logic_provider"
)


class TacticianHammerCapabilityStatus(str, Enum):
    """Closed admission outcomes; only ``AVAILABLE`` admits an interface."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    UNSUPPORTED = "unsupported"
    INCOMPATIBLE = "incompatible"
    PARTIAL = "partial"
    TIMED_OUT = "timed_out"


class TacticianHammerDiagnosticCode(str, Enum):
    """Machine-readable reason for a non-admitted capability."""

    MODULE_IMPORT_FAILED = "module_import_failed"
    MODULE_PATH_UNAVAILABLE = "module_path_unavailable"
    REQUIRED_SYMBOL_MISSING = "required_symbol_missing"
    REQUIRED_SIGNATURE_INCOMPATIBLE = "required_signature_incompatible"
    INTERFACE_VERSION_MISSING = "interface_version_missing"
    INTERFACE_VERSION_INCOMPATIBLE = "interface_version_incompatible"
    SCHEMA_VERSION_MISSING = "schema_version_missing"
    SCHEMA_VERSION_INCOMPATIBLE = "schema_version_incompatible"
    PARTIAL_INTERFACE = "partial_interface"
    PROBE_TIMED_OUT = "probe_timed_out"
    EXECUTABLE_NOT_FOUND = "executable_not_found"
    EXECUTABLE_VERSION_FAILED = "executable_version_failed"
    EXECUTABLE_VERSION_INCOMPATIBLE = "executable_version_incompatible"
    GITLINK_UNAVAILABLE = "gitlink_unavailable"
    GITLINK_MALFORMED = "gitlink_malformed"
    PENDING_LPR_003 = "pending_lpr_003"
    LEGAL_ADAPTER_ONLY = "legal_adapter_only"
    IMPORT_ISOLATION_UNSAFE = "import_isolation_unsafe"
    FEATURE_NOT_ADMITTED = "feature_not_admitted"
    INTERNAL_ERROR = "internal_error"


class ResourceEnforcementStrength(str, Enum):
    """How strongly the host can enforce CPU/memory process bounds."""

    POSIX_RLIMIT = "posix_rlimit"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class TacticianHammerCapabilityDiagnostic:
    """Typed, bounded probe diagnostic rather than an exception-only failure."""

    code: TacticianHammerDiagnosticCode
    capability_id: str
    message: str
    module: str = ""
    exception_type: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code.value,
            "capability_id": self.capability_id,
            "message": self.message,
            "module": self.module,
            "exception_type": self.exception_type,
        }


@dataclass(frozen=True)
class TacticianHammerCapability:
    """One exact capability binding used by Tactician-Hammer admission."""

    capability_id: str
    status: TacticianHammerCapabilityStatus
    module_paths: tuple[str, ...] = ()
    interface_version: str = ""
    schema_version: str = ""
    producer_id: str = ""
    operations: tuple[str, ...] = ()
    supported_semantics: tuple[str, ...] = ()
    diagnostic: TacticianHammerCapabilityDiagnostic | None = None
    reconstruction_compatible: bool = False
    candidate_authoritative: bool = False
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.capability_id.strip():
            raise ValueError("capability_id must not be empty")
        if self.status is TacticianHammerCapabilityStatus.AVAILABLE:
            if not self.module_paths and not self.details.get("executable_path"):
                raise ValueError(
                    "available capability requires an exact module or executable path"
                )
            if self.diagnostic is not None:
                raise ValueError("available capability cannot carry a failure diagnostic")
        elif self.diagnostic is None:
            raise ValueError("non-available capability requires a typed diagnostic")
        if self.candidate_authoritative:
            raise ValueError(
                "solver, graph, vector, and model candidates cannot be authoritative"
            )
        object.__setattr__(self, "module_paths", tuple(sorted(set(self.module_paths))))
        object.__setattr__(self, "operations", tuple(sorted(set(self.operations))))
        object.__setattr__(
            self, "supported_semantics", tuple(sorted(set(self.supported_semantics)))
        )
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))

    @property
    def available(self) -> bool:
        return self.status is TacticianHammerCapabilityStatus.AVAILABLE

    @property
    def module_path(self) -> str:
        """Primary exact path, retained for single-module consumers."""

        return self.module_paths[0] if self.module_paths else ""

    @property
    def reason_code(self) -> str:
        return self.diagnostic.code.value if self.diagnostic is not None else ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability_id": self.capability_id,
            "status": self.status.value,
            "available": self.available,
            "module_paths": list(self.module_paths),
            "interface_version": self.interface_version,
            "schema_version": self.schema_version,
            "producer_id": self.producer_id,
            "operations": list(self.operations),
            "supported_semantics": list(self.supported_semantics),
            "reconstruction_compatible": self.reconstruction_compatible,
            "candidate_authoritative": False,
            "diagnostic": self.diagnostic.to_dict() if self.diagnostic else None,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class ResourceEnforcementReport:
    """Platform-typed resource-enforcement strength for native solver lanes."""

    platform: str
    cpu_enforcement: ResourceEnforcementStrength
    memory_enforcement: ResourceEnforcementStrength
    process_isolation: ResourceEnforcementStrength
    network_policy_denied: bool = True
    network_os_isolation: bool = False
    environment_lock_path_version_only: bool = True
    signed_binary_integrity: bool = False
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "platform": self.platform,
            "cpu_enforcement": self.cpu_enforcement.value,
            "memory_enforcement": self.memory_enforcement.value,
            "process_isolation": self.process_isolation.value,
            "network_policy_denied": self.network_policy_denied,
            "network_os_isolation": self.network_os_isolation,
            "environment_lock_path_version_only": self.environment_lock_path_version_only,
            "signed_binary_integrity": self.signed_binary_integrity,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class TacticianHammerCapabilityReport:
    """Immutable, versioned snapshot of Tactician-Hammer prerequisites."""

    capabilities: tuple[TacticianHammerCapability, ...]
    accelerator_module_paths: tuple[str, ...]
    datasets_module_paths: tuple[str, ...]
    datasets_gitlink_revision: str
    import_isolation: str = IMPORT_ISOLATION_UNSAFE
    resource_enforcement: ResourceEnforcementReport | None = None
    diagnostics: tuple[TacticianHammerCapabilityDiagnostic, ...] = ()
    generated_at_monotonic: float = 0.0
    duration_seconds: float = 0.0
    schema_version: str = TACTICIAN_HAMMER_CAPABILITY_REPORT_SCHEMA_VERSION
    report_version: int = TACTICIAN_HAMMER_CAPABILITY_REPORT_VERSION
    # Feature admissions default fail-closed.
    learned_selector_admitted: bool = False
    model_execution_admitted: bool = False
    native_execution_admitted: bool = False
    network_access_admitted: bool = False
    auto_install_admitted: bool = False

    def __post_init__(self) -> None:
        ids = [item.capability_id for item in self.capabilities]
        if len(ids) != len(set(ids)):
            raise ValueError("capability ids must be unique")
        if self.report_version != TACTICIAN_HAMMER_CAPABILITY_REPORT_VERSION:
            raise ValueError("unsupported capability report version")
        if self.duration_seconds < 0:
            raise ValueError("duration_seconds must be non-negative")
        if self.import_isolation not in {
            IMPORT_ISOLATION_UNSAFE,
            IMPORT_ISOLATION_HARDENED,
        }:
            raise ValueError("import_isolation must be a known isolation state")
        object.__setattr__(self, "capabilities", tuple(self.capabilities))
        object.__setattr__(
            self, "accelerator_module_paths", tuple(sorted(set(self.accelerator_module_paths)))
        )
        object.__setattr__(
            self, "datasets_module_paths", tuple(sorted(set(self.datasets_module_paths)))
        )
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))

    @property
    def capability_map(self) -> Mapping[str, TacticianHammerCapability]:
        return MappingProxyType({item.capability_id: item for item in self.capabilities})

    @property
    def capabilities_by_id(self) -> Mapping[str, TacticianHammerCapability]:
        return self.capability_map

    def capability(self, capability_id: str) -> TacticianHammerCapability:
        try:
            return self.capability_map[capability_id]
        except KeyError as exc:
            raise KeyError(
                f"unknown tactician-hammer capability: {capability_id}"
            ) from exc

    @property
    def toolchains(self) -> Mapping[str, TacticianHammerCapability]:
        return MappingProxyType(
            {
                item.capability_id: item
                for item in self.capabilities
                if item.capability_id.startswith("toolchain.")
            }
        )

    @property
    def gitlink_revision(self) -> str:
        return self.datasets_gitlink_revision

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "report_version": self.report_version,
            "duration_seconds": self.duration_seconds,
            "accelerator_module_paths": list(self.accelerator_module_paths),
            "datasets_module_paths": list(self.datasets_module_paths),
            "datasets_gitlink_revision": self.datasets_gitlink_revision,
            "import_isolation": self.import_isolation,
            "resource_enforcement": (
                self.resource_enforcement.to_dict()
                if self.resource_enforcement is not None
                else None
            ),
            "capabilities": {
                item.capability_id: item.to_dict() for item in self.capabilities
            },
            "toolchains": {
                item.capability_id: item.to_dict() for item in self.toolchains.values()
            },
            "diagnostics": [item.to_dict() for item in self.diagnostics],
            # Fail-closed admissions.  Explicit report fields, never inferred.
            "network_access": False,
            "network_access_admitted": self.network_access_admitted,
            "auto_install": False,
            "auto_install_admitted": self.auto_install_admitted,
            "learned_selector_admitted": self.learned_selector_admitted,
            "model_execution_admitted": self.model_execution_admitted,
            "native_execution_admitted": self.native_execution_admitted,
            "solver_candidates_authoritative": False,
            "vector_semantic_authority": False,
            "graph_semantic_authority": False,
            "llm_completion_authority": False,
            "tactician_proof_authority": False,
            "network_policy_denied_is_os_isolation": False,
            "environment_lock_is_signed_binary_integrity": False,
            "legal_tactician_disposition": "legal_adapter_only",
        }


@dataclass(frozen=True)
class _InterfaceSpec:
    capability_id: str
    module: str
    symbols: tuple[str, ...]
    interface_constant: str = ""
    expected_interface: str = ""
    schema_constant: str = ""
    expected_schema: str = ""
    producer_id: str = ""
    operations: tuple[str, ...] = ()
    semantics: tuple[str, ...] = ()
    reconstruction_compatible: bool = False
    # When True, package/module presence alone yields PARTIAL, not AVAILABLE,
    # unless every listed symbol is bound.  Always enforced by the probe.
    require_all_symbols: bool = True


# Static-analysis, vector, KG, and llm surfaces.  Package markers never fill
# missing symbols.
_STATIC_AND_AUX_SPECS: Final = (
    _InterfaceSpec(
        "analyzer.ast",
        "ipfs_accelerate_py.agent_supervisor.analysis.analysis_ast_index",
        ("AnalysisASTIndex", "build_analysis_ast_index"),
        schema_constant="ANALYSIS_AST_INDEX_SCHEMA",
        producer_id="analysis-ast-index@1",
        operations=("ast_index", "ast_query"),
        semantics=("analysis_ast_index", "snapshot_bound"),
    ),
    _InterfaceSpec(
        "analyzer.call",
        "ipfs_accelerate_py.agent_supervisor.program_call_resolver",
        ("ProgramCallResolver",),
        interface_constant="PROGRAM_CALL_RESOLVER_VERSION",
        schema_constant="PROGRAM_CALL_RESOLVER_SCHEMA",
        producer_id="program-call-resolver@1",
        operations=("resolve_call", "call_resolution"),
        semantics=("conservative_call_resolution", "unknown_frontier"),
    ),
    _InterfaceSpec(
        "analyzer.dataflow",
        "ipfs_accelerate_py.agent_supervisor.analysis.code_evidence_graph",
        ("CodeEvidenceGraph", "ProvenanceEdge", "CodeImpactIndex"),
        schema_constant="CODE_EVIDENCE_GRAPH_SCHEMA",
        producer_id="code-evidence-graph@1",
        operations=("dataflow", "value_flow", "impact_selection"),
        semantics=("dataflow", "value_provenance", "impact_graph"),
    ),
    _InterfaceSpec(
        "analyzer.type",
        "ipfs_accelerate_py.agent_supervisor.program_contracts",
        ("ExpectedProgramContract", "ObservedProgramContract", "ProgramContractBundle"),
        interface_constant="PROGRAM_CONTRACT_VERSION",
        expected_interface="1",
        schema_constant="SCHEMA_VERSION",
        expected_schema="1",
        producer_id="program-contracts@1",
        operations=("type_contract", "program_contract"),
        semantics=("ProgramContract@1", "static_type_surface"),
    ),
    _InterfaceSpec(
        "analyzer.effect",
        "ipfs_accelerate_py.agent_supervisor.analysis.memory_safety_facets",
        ("MemorySafetyEvidenceCollector", "MemorySafetyPolicy", "NativeBoundary"),
        schema_constant="MEMORY_SAFETY_EVIDENCE_SCHEMA",
        producer_id="memory-safety-facets@1",
        operations=("effect_facet", "memory_safety", "native_boundary"),
        semantics=("MemorySafetyFacet@1", "effect_analysis", "scoped_memory_claims"),
    ),
    _InterfaceSpec(
        "analyzer.program_graph",
        "ipfs_accelerate_py.agent_supervisor.program_graph",
        ("ProgramGraph", "ProgramGraphSnapshot"),
        interface_constant="PROGRAM_GRAPH_VERSION",
        schema_constant="PROGRAM_GRAPH_SCHEMA",
        producer_id="program-graph@1",
        operations=("trace_graph_evidence", "call_edges", "data_edges"),
        semantics=("program_graph", "complete_frontier", "unknown_frontier"),
    ),
    _InterfaceSpec(
        "vector.code_symbol",
        "ipfs_accelerate_py.agent_supervisor.analysis.code_symbol_vector_index",
        (
            "CodeSymbolVectorIndex",
            "build_code_symbol_vector_index",
            "search_code_symbol_vector_index",
        ),
        schema_constant="CODE_SYMBOL_VECTOR_INDEX_SCHEMA",
        producer_id="code-symbol-vector-index@1",
        operations=("vector_build", "vector_search"),
        semantics=(
            "code_symbol_vector_index",
            "semantic_authority_false",
            "nomination_only",
            "stale_safe",
        ),
    ),
    _InterfaceSpec(
        "vector.change_value",
        "ipfs_accelerate_py.agent_supervisor.analysis.change_value_vector_index",
        ("ChangeValueVectorIndex", "CHANGE_VALUE_VECTOR_INDEX_SCHEMA"),
        schema_constant="CHANGE_VALUE_VECTOR_INDEX_SCHEMA",
        producer_id="change-value-vector-index@1",
        operations=("vector_build", "vector_search"),
        semantics=(
            "change_value_vector_index",
            "semantic_authority_false",
            "nomination_only",
        ),
    ),
    _InterfaceSpec(
        "datasets.logic_provider",
        LOGIC_PROVIDER_MODULE,
        (
            "IpfsDatasetsLogicProvider",
            "DatasetsLogicBackendProbe",
            "probe_all_datasets_logic_backends",
        ),
        interface_constant="IPFS_DATASETS_LOGIC_PROVIDER_VERSION",
        schema_constant="HAMMER_ADAPTER_SCHEMA_VERSION",
        producer_id="ipfs-datasets-logic-provider@1",
        operations=("logic_ir", "hammer", "reconstruction"),
        semantics=(
            "IPFSDatasetsLogicProvider",
            "BackendCapability",
            "solver_candidates_non_authoritative",
            "independent_reconstruction_required",
        ),
        reconstruction_compatible=True,
    ),
    _InterfaceSpec(
        "datasets.analysis_provider",
        "ipfs_accelerate_py.agent_supervisor.integrations.ipfs_datasets_analysis_provider",
        (
            "IpfsDatasetsAnalysisProvider",
            "AnalysisProviderOperation",
            "probe_all_datasets_graph_backends",
        ),
        producer_id="ipfs-datasets-analysis-provider@1",
        operations=("graph_retrieval", "premise_selection", "symbol_impact"),
        semantics=(
            "IPFSDatasetsAnalysisProvider",
            "GraphRAG_non_authoritative",
            "nomination_only",
        ),
    ),
    _InterfaceSpec(
        "llm.router",
        "ipfs_accelerate_py.llm_router",
        ("generate_text", "get_last_usage_admission", "get_last_generation_trace"),
        producer_id="llm-router@1",
        operations=("text.generate", "text.chat"),
        semantics=(
            "llm_router",
            "proposal_only",
            "non_authoritative",
            "no_completion_authority",
        ),
    ),
)

# Exact Hammer descriptor surfaces.  Submodule imports avoid calling the
# supervisor's process-global HOME/sys.prefix lazy loader.
_HAMMER_SPECS: Final = (
    _InterfaceSpec(
        "hammer.corpus",
        "ipfs_datasets_py.logic.hammers.corpus",
        ("CorpusManifest", "TheoremEntry", "CorpusSource"),
        producer_id="hammer-corpus@1",
        operations=("corpus_manifest", "theorem_entry"),
        semantics=("content_addressed_corpus", "corpus_revision"),
    ),
    _InterfaceSpec(
        "hammer.selector.deterministic",
        "ipfs_datasets_py.logic.hammers.premise_selection",
        ("select_premises", "GoalFeatures", "PremiseSelectionResult"),
        producer_id="hammer-deterministic-selector@1",
        operations=("select_premises", "deterministic_ranking"),
        semantics=("deterministic_premise_selection", "default_selector"),
    ),
    _InterfaceSpec(
        "hammer.selector.learned",
        "ipfs_datasets_py.logic.hammers.learned_selector",
        (
            "select_premises_gated",
            "LearnedModelArtifact",
            "LearnedSelectorConfig",
            "SelectorFallbackReason",
        ),
        producer_id="hammer-learned-selector@1",
        operations=("select_premises_gated", "learned_ranking"),
        semantics=(
            "learned_premise_selection",
            "opt_in_only",
            "deterministic_fallback",
            "ranking_only",
            "feature_not_admitted_by_default",
        ),
    ),
    _InterfaceSpec(
        "hammer.translation",
        "ipfs_datasets_py.logic.hammers.models",
        ("TranslationTarget", "TranslationRecord", "TranslationStatus"),
        interface_constant="SCHEMA_VERSION",
        producer_id="hammer-translation@1",
        operations=("translate_tptp", "translate_smtlib"),
        semantics=("tptp", "smtlib", "translation_map_required"),
    ),
    _InterfaceSpec(
        "hammer.translation.map",
        "ipfs_datasets_py.logic.hammers.translation",
        ("TranslationMap", "TranslationMapEntry", "TranslationContext"),
        producer_id="hammer-translation-map@1",
        operations=("translation_map", "lower_construct"),
        semantics=("translation_map", "unsupported_construct_explicit"),
    ),
    _InterfaceSpec(
        "hammer.portfolio",
        "ipfs_datasets_py.logic.hammers.portfolio",
        ("SolverPortfolio", "PortfolioRunResult", "SolverAttemptEvidence"),
        producer_id="hammer-portfolio@1",
        operations=("solver_portfolio", "bounded_attempt"),
        semantics=(
            "z3",
            "cvc5",
            "vampire",
            "e",
            "solver_candidates_non_authoritative",
        ),
    ),
    _InterfaceSpec(
        "hammer.receipt",
        "ipfs_datasets_py.logic.hammers.receipts",
        ("HammerReceipt", "ReceiptStore", "compute_receipt_digest"),
        producer_id="hammer-receipt@1",
        operations=("hammer_receipt", "receipt_store", "publishable_view"),
        semantics=("content_addressed_receipt", "replayable"),
    ),
    _InterfaceSpec(
        "hammer.environment_lock",
        "ipfs_datasets_py.logic.hammers.models",
        ("EnvironmentLockRecord",),
        interface_constant="SCHEMA_VERSION",
        producer_id="hammer-environment-lock@1",
        operations=("environment_lock",),
        semantics=(
            "path_version_lock",
            "not_signed_binary_integrity",
            "environment_binding",
        ),
    ),
    _InterfaceSpec(
        "hammer.reconstruction.lean",
        "ipfs_datasets_py.logic.hammers.reconstructors.lean",
        ("LeanReconstructor",),
        producer_id="hammer-reconstruction-lean@1",
        operations=("reconstruct_lean", "kernel_check"),
        semantics=("lean", "kernel_reconstruction", "independent_reconstruction_required"),
        reconstruction_compatible=True,
    ),
    _InterfaceSpec(
        "hammer.reconstruction.coq",
        "ipfs_datasets_py.logic.hammers.reconstructors.coq",
        ("CoqReconstructor",),
        producer_id="hammer-reconstruction-coq@1",
        operations=("reconstruct_coq", "kernel_check"),
        semantics=("coq", "kernel_reconstruction", "independent_reconstruction_required"),
        reconstruction_compatible=True,
    ),
    _InterfaceSpec(
        "hammer.reconstruction.isabelle",
        "ipfs_datasets_py.logic.hammers.reconstructors.isabelle",
        ("IsabelleReconstructor",),
        producer_id="hammer-reconstruction-isabelle@1",
        operations=("reconstruct_isabelle", "kernel_check"),
        semantics=(
            "isabelle",
            "kernel_reconstruction",
            "independent_reconstruction_required",
        ),
        reconstruction_compatible=True,
    ),
    _InterfaceSpec(
        "hammer.reconstruction.api",
        "ipfs_datasets_py.logic.hammers.reconstruction",
        ("reconstruct_candidate", "get_reconstructor", "build_environment_lock"),
        producer_id="hammer-reconstruction-api@1",
        operations=("reconstruct_candidate", "environment_lock"),
        semantics=("independent_reconstruction_required", "kernel_checked_only_verified"),
        reconstruction_compatible=True,
    ),
)


def _diagnostic(
    code: TacticianHammerDiagnosticCode,
    capability_id: str,
    message: str,
    *,
    module: str = "",
    exception: BaseException | None = None,
) -> TacticianHammerCapabilityDiagnostic:
    return TacticianHammerCapabilityDiagnostic(
        code=code,
        capability_id=capability_id,
        message=message,
        module=module,
        exception_type=type(exception).__name__ if exception else "",
    )


def _bounded_call(
    callback: Callable[[], Any], timeout_seconds: float
) -> tuple[bool, Any, BaseException | None]:
    """Run an import/probe under a wall clock budget without killing callers."""

    result: list[Any] = []
    error: list[BaseException] = []

    def invoke() -> None:
        try:
            result.append(callback())
        except BaseException as exc:  # import hooks may use non-Exception errors
            error.append(exc)

    worker = threading.Thread(target=invoke, daemon=True)
    worker.start()
    worker.join(timeout_seconds)
    if worker.is_alive():
        return False, None, None
    return True, (result[0] if result else None), (error[0] if error else None)


def _module_path(module: Any) -> str:
    path = getattr(module, "__file__", "")
    if not isinstance(path, str) or not path:
        return ""
    return os.path.realpath(path)


def _probe_interface(
    spec: _InterfaceSpec,
    *,
    importer: Callable[[str], Any],
    timeout_seconds: float,
) -> TacticianHammerCapability:
    completed, module, error = _bounded_call(
        lambda: importer(spec.module), timeout_seconds
    )
    if not completed:
        return TacticianHammerCapability(
            spec.capability_id,
            TacticianHammerCapabilityStatus.TIMED_OUT,
            producer_id=spec.producer_id,
            operations=spec.operations,
            diagnostic=_diagnostic(
                TacticianHammerDiagnosticCode.PROBE_TIMED_OUT,
                spec.capability_id,
                "module import exceeded probe timeout",
                module=spec.module,
            ),
        )
    if error is not None:
        return TacticianHammerCapability(
            spec.capability_id,
            TacticianHammerCapabilityStatus.UNAVAILABLE,
            producer_id=spec.producer_id,
            operations=spec.operations,
            diagnostic=_diagnostic(
                TacticianHammerDiagnosticCode.MODULE_IMPORT_FAILED,
                spec.capability_id,
                "required module could not be imported",
                module=spec.module,
                exception=error,
            ),
        )
    path = _module_path(module)
    if not path:
        return TacticianHammerCapability(
            spec.capability_id,
            TacticianHammerCapabilityStatus.INCOMPATIBLE,
            producer_id=spec.producer_id,
            operations=spec.operations,
            diagnostic=_diagnostic(
                TacticianHammerDiagnosticCode.MODULE_PATH_UNAVAILABLE,
                spec.capability_id,
                "imported module has no exact file path",
                module=spec.module,
            ),
        )
    missing = [symbol for symbol in spec.symbols if not hasattr(module, symbol)]
    if missing:
        # Package/module presence alone is insufficient for admission.
        return TacticianHammerCapability(
            spec.capability_id,
            TacticianHammerCapabilityStatus.PARTIAL,
            module_paths=(path,),
            producer_id=spec.producer_id,
            operations=spec.operations,
            diagnostic=_diagnostic(
                TacticianHammerDiagnosticCode.REQUIRED_SYMBOL_MISSING,
                spec.capability_id,
                f"missing required interface symbols: {', '.join(missing)}",
                module=spec.module,
            ),
            details={"package_present": True, "missing_symbols": list(missing)},
        )
    interface_version = (
        str(getattr(module, spec.interface_constant, ""))
        if spec.interface_constant
        else ""
    )
    if spec.expected_interface and str(interface_version) != str(spec.expected_interface):
        return TacticianHammerCapability(
            spec.capability_id,
            TacticianHammerCapabilityStatus.INCOMPATIBLE,
            module_paths=(path,),
            interface_version=interface_version,
            producer_id=spec.producer_id,
            operations=spec.operations,
            diagnostic=_diagnostic(
                TacticianHammerDiagnosticCode.INTERFACE_VERSION_INCOMPATIBLE,
                spec.capability_id,
                f"expected interface version {spec.expected_interface!r}, "
                f"got {interface_version!r}",
                module=spec.module,
            ),
        )
    schema_version = (
        str(getattr(module, spec.schema_constant, "")) if spec.schema_constant else ""
    )
    if spec.expected_schema and str(schema_version) != str(spec.expected_schema):
        return TacticianHammerCapability(
            spec.capability_id,
            TacticianHammerCapabilityStatus.INCOMPATIBLE,
            module_paths=(path,),
            interface_version=interface_version,
            schema_version=schema_version,
            producer_id=spec.producer_id,
            operations=spec.operations,
            diagnostic=_diagnostic(
                TacticianHammerDiagnosticCode.SCHEMA_VERSION_INCOMPATIBLE,
                spec.capability_id,
                f"expected schema version {spec.expected_schema!r}, "
                f"got {schema_version!r}",
                module=spec.module,
            ),
        )
    return TacticianHammerCapability(
        spec.capability_id,
        TacticianHammerCapabilityStatus.AVAILABLE,
        module_paths=(path,),
        interface_version=interface_version,
        schema_version=schema_version,
        producer_id=spec.producer_id,
        operations=spec.operations,
        supported_semantics=spec.semantics,
        reconstruction_compatible=spec.reconstruction_compatible,
    )


def _probe_generic_tactician(
    *, importer: Callable[[str], Any], timeout_seconds: float
) -> TacticianHammerCapability:
    """Probe domain-neutral Logic Tactician; unavailable until LPR-003 lands it."""

    capability_id = "tactician.generic"
    # Prefer find_spec before import so a missing package does not pull
    # optional side-effectful package roots when an importer is the default.
    if importer is importlib.import_module:
        try:
            spec = importlib.util.find_spec(GENERIC_TACTICIAN_MODULE)
        except (ImportError, ModuleNotFoundError, ValueError):
            spec = None
        if spec is None:
            return TacticianHammerCapability(
                capability_id,
                TacticianHammerCapabilityStatus.UNAVAILABLE,
                producer_id="logic-tactician@pending",
                operations=("plan", "decompose", "nominate"),
                diagnostic=_diagnostic(
                    TacticianHammerDiagnosticCode.PENDING_LPR_003,
                    capability_id,
                    "domain-neutral Logic Tactician is not present; pending LPR-003",
                    module=GENERIC_TACTICIAN_MODULE,
                ),
                details={
                    "pending_task": "LPR-003",
                    "expected_module": GENERIC_TACTICIAN_MODULE,
                    "expected_symbols": list(GENERIC_TACTICIAN_SYMBOLS),
                    "domain_neutral": True,
                },
            )

    completed, module, error = _bounded_call(
        lambda: importer(GENERIC_TACTICIAN_MODULE), timeout_seconds
    )
    if not completed:
        return TacticianHammerCapability(
            capability_id,
            TacticianHammerCapabilityStatus.TIMED_OUT,
            producer_id="logic-tactician@pending",
            operations=("plan", "decompose", "nominate"),
            diagnostic=_diagnostic(
                TacticianHammerDiagnosticCode.PROBE_TIMED_OUT,
                capability_id,
                "generic Tactician import exceeded probe timeout",
                module=GENERIC_TACTICIAN_MODULE,
            ),
            details={"pending_task": "LPR-003", "domain_neutral": True},
        )
    if error is not None:
        return TacticianHammerCapability(
            capability_id,
            TacticianHammerCapabilityStatus.UNAVAILABLE,
            producer_id="logic-tactician@pending",
            operations=("plan", "decompose", "nominate"),
            diagnostic=_diagnostic(
                TacticianHammerDiagnosticCode.PENDING_LPR_003,
                capability_id,
                "domain-neutral Logic Tactician is not present; pending LPR-003",
                module=GENERIC_TACTICIAN_MODULE,
                exception=error,
            ),
            details={
                "pending_task": "LPR-003",
                "expected_module": GENERIC_TACTICIAN_MODULE,
                "expected_symbols": list(GENERIC_TACTICIAN_SYMBOLS),
                "domain_neutral": True,
            },
        )
    path = _module_path(module)
    missing = [symbol for symbol in GENERIC_TACTICIAN_SYMBOLS if not hasattr(module, symbol)]
    if missing or not path:
        return TacticianHammerCapability(
            capability_id,
            TacticianHammerCapabilityStatus.PARTIAL
            if path
            else TacticianHammerCapabilityStatus.UNAVAILABLE,
            module_paths=(path,) if path else (),
            producer_id="logic-tactician@1",
            operations=("plan", "decompose", "nominate"),
            diagnostic=_diagnostic(
                TacticianHammerDiagnosticCode.PARTIAL_INTERFACE
                if path
                else TacticianHammerDiagnosticCode.MODULE_PATH_UNAVAILABLE,
                capability_id,
                (
                    f"generic Tactician present but missing symbols: {', '.join(missing)}"
                    if missing
                    else "generic Tactician module has no exact file path"
                ),
                module=GENERIC_TACTICIAN_MODULE,
            ),
            details={
                "pending_task": "LPR-003",
                "package_present": True,
                "missing_symbols": list(missing),
                "domain_neutral": True,
            },
        )
    interface_version = str(
        getattr(module, "LOGIC_TACTICIAN_INTERFACE", getattr(module, "SCHEMA_VERSION", ""))
    )
    schema_version = str(
        getattr(module, "TACTICIAN_SCHEMA_VERSION", getattr(module, "SCHEMA_VERSION", ""))
    )
    return TacticianHammerCapability(
        capability_id,
        TacticianHammerCapabilityStatus.AVAILABLE,
        module_paths=(path,),
        interface_version=interface_version,
        schema_version=schema_version,
        producer_id="logic-tactician@1",
        operations=("plan", "decompose", "nominate", "receipt"),
        supported_semantics=(
            "domain_neutral",
            "advisory_only",
            "no_proof_authority",
            "no_write_authority",
            "finite_acyclic_plan",
        ),
        details={"domain_neutral": True, "pending_task": ""},
    )


def _probe_legal_tactician_adapter(
    *, importer: Callable[[str], Any], timeout_seconds: float
) -> TacticianHammerCapability:
    """Retain legal ProofTactician as legal-adapter-only, never code authority."""

    capability_id = "tactician.legal_adapter"
    completed, module, error = _bounded_call(
        lambda: importer(LEGAL_TACTICIAN_MODULE), timeout_seconds
    )
    if not completed:
        return TacticianHammerCapability(
            capability_id,
            TacticianHammerCapabilityStatus.TIMED_OUT,
            producer_id="legal-proof-tactician@adapter",
            operations=("legal_search_plan",),
            diagnostic=_diagnostic(
                TacticianHammerDiagnosticCode.PROBE_TIMED_OUT,
                capability_id,
                "legal Tactician import exceeded probe timeout",
                module=LEGAL_TACTICIAN_MODULE,
            ),
            details={"disposition": "legal_adapter_only", "code_authority": False},
        )
    if error is not None:
        return TacticianHammerCapability(
            capability_id,
            TacticianHammerCapabilityStatus.UNAVAILABLE,
            producer_id="legal-proof-tactician@adapter",
            operations=("legal_search_plan",),
            diagnostic=_diagnostic(
                TacticianHammerDiagnosticCode.MODULE_IMPORT_FAILED,
                capability_id,
                "legal ProofTactician adapter module could not be imported",
                module=LEGAL_TACTICIAN_MODULE,
                exception=error,
            ),
            details={"disposition": "legal_adapter_only", "code_authority": False},
        )
    path = _module_path(module)
    missing = [symbol for symbol in LEGAL_TACTICIAN_SYMBOLS if not hasattr(module, symbol)]
    if missing or not path:
        return TacticianHammerCapability(
            capability_id,
            TacticianHammerCapabilityStatus.PARTIAL
            if path
            else TacticianHammerCapabilityStatus.UNAVAILABLE,
            module_paths=(path,) if path else (),
            producer_id="legal-proof-tactician@adapter",
            operations=("legal_search_plan",),
            diagnostic=_diagnostic(
                TacticianHammerDiagnosticCode.REQUIRED_SYMBOL_MISSING
                if missing
                else TacticianHammerDiagnosticCode.MODULE_PATH_UNAVAILABLE,
                capability_id,
                (
                    f"missing legal adapter symbols: {', '.join(missing)}"
                    if missing
                    else "legal Tactician module has no exact file path"
                ),
                module=LEGAL_TACTICIAN_MODULE,
            ),
            details={
                "disposition": "legal_adapter_only",
                "code_authority": False,
                "package_present": bool(path),
            },
        )
    # Available only as a legal domain adapter.  Explicitly not code-repair
    # authority; callers must not promote this surface into program logic.
    return TacticianHammerCapability(
        capability_id,
        TacticianHammerCapabilityStatus.AVAILABLE,
        module_paths=(path,),
        producer_id="legal-proof-tactician@adapter",
        operations=("legal_search_plan", "legal_proof_gap_focus"),
        supported_semantics=(
            "legal_adapter_only",
            "not_code_authority",
            "domain_legal",
            "ordered_search_plan_pattern",
        ),
        details={
            "disposition": "legal_adapter_only",
            "code_authority": False,
            "proof_authority": False,
            "write_authority": False,
            "domain": "legal",
        },
    )


def _probe_import_isolation(
    *, importer: Callable[[str], Any], timeout_seconds: float
) -> tuple[TacticianHammerCapability, str]:
    """Report process-global HOME/sys.prefix lazy-load as unsafe until LPR-012.

    Hardened isolation is fail-closed: only an explicit provider declaration of
    ``HAMMER_IMPORT_ISOLATION == import_isolation_hardened`` (set by LPR-012)
    upgrades the report.  Source inspection may refine ``mutates_*`` details but
    never optimistically claims hardening from a missing mutation pattern.
    """

    capability_id = "hammer.import_isolation"
    isolation_state = IMPORT_ISOLATION_UNSAFE
    details: dict[str, Any] = {
        "pending_task": "LPR-012",
        "mutates_home": True,
        "mutates_sys_prefix": True,
        "process_global": True,
        "concurrency_safe": False,
    }
    completed, module, error = _bounded_call(
        lambda: importer(LOGIC_PROVIDER_MODULE), timeout_seconds
    )
    module_paths: tuple[str, ...] = ()
    if completed and error is None and module is not None:
        path = _module_path(module)
        if path:
            module_paths = (path,)
        declared = str(getattr(module, "HAMMER_IMPORT_ISOLATION", "") or "")
        details["declared_isolation"] = declared
        loader = getattr(module, "_load_hammer", None)
        if callable(loader):
            try:
                source = inspect.getsource(loader)
            except (OSError, TypeError):
                source = ""
            if source:
                # Only refine details when mutation is observed.  Absence of a
                # pattern in a thin wrapper must not clear the unsafe default.
                if "HOME" in source and (
                    "os.environ" in source or "environ[" in source
                ):
                    details["mutates_home"] = True
                if "sys.prefix" in source:
                    details["mutates_sys_prefix"] = True
                details["load_function"] = "_load_hammer"
                details["source_inspected"] = True
        else:
            details["load_function"] = ""
            details["note"] = "logic provider present without _load_hammer symbol"
        # Positive hardening signal only (LPR-012).  Never infer from silence.
        if declared == IMPORT_ISOLATION_HARDENED:
            isolation_state = IMPORT_ISOLATION_HARDENED
            details["pending_task"] = ""
            details["concurrency_safe"] = True
            details["mutates_home"] = False
            details["mutates_sys_prefix"] = False
            details["process_global"] = False
    else:
        details["logic_provider_importable"] = False
        if error is not None:
            details["exception_type"] = type(error).__name__

    if isolation_state == IMPORT_ISOLATION_UNSAFE:
        capability = TacticianHammerCapability(
            capability_id,
            TacticianHammerCapabilityStatus.PARTIAL,
            module_paths=module_paths,
            producer_id="hammer-import-isolation@1",
            operations=("lazy_hammer_import",),
            diagnostic=_diagnostic(
                TacticianHammerDiagnosticCode.IMPORT_ISOLATION_UNSAFE,
                capability_id,
                "current Hammer lazy-load mutates process-global HOME/sys.prefix; "
                "reported import_isolation_unsafe until LPR-012 hardens it",
                module=LOGIC_PROVIDER_MODULE,
            ),
            details=details,
            supported_semantics=(),
        )
    else:
        if not module_paths:
            details = {**details, "executable_path": LOGIC_PROVIDER_MODULE}
        capability = TacticianHammerCapability(
            capability_id,
            TacticianHammerCapabilityStatus.AVAILABLE,
            module_paths=module_paths,
            producer_id="hammer-import-isolation@1",
            operations=("lazy_hammer_import",),
            supported_semantics=("import_isolation_hardened", "concurrency_safe"),
            details=details,
        )
    return capability, isolation_state


def _probe_kg_provider(
    *, importer: Callable[[str], Any], timeout_seconds: float
) -> TacticianHammerCapability:
    """Probe GraphRAG / KG nomination surface without granting graph authority."""

    capability_id = "kg.graphrag"
    provider_module = (
        "ipfs_accelerate_py.agent_supervisor.integrations.ipfs_datasets_analysis_provider"
    )
    completed, provider, error = _bounded_call(
        lambda: importer(provider_module), timeout_seconds
    )
    if not completed:
        return TacticianHammerCapability(
            capability_id,
            TacticianHammerCapabilityStatus.TIMED_OUT,
            producer_id="ipfs-datasets-analysis-provider@1",
            operations=("graph_retrieval", "kg_nomination"),
            diagnostic=_diagnostic(
                TacticianHammerDiagnosticCode.PROBE_TIMED_OUT,
                capability_id,
                "KG provider import exceeded probe timeout",
                module=provider_module,
            ),
        )
    if error is not None:
        return TacticianHammerCapability(
            capability_id,
            TacticianHammerCapabilityStatus.UNAVAILABLE,
            producer_id="ipfs-datasets-analysis-provider@1",
            operations=("graph_retrieval", "kg_nomination"),
            diagnostic=_diagnostic(
                TacticianHammerDiagnosticCode.MODULE_IMPORT_FAILED,
                capability_id,
                "KG analysis provider could not be imported",
                module=provider_module,
                exception=error,
            ),
        )
    path = _module_path(provider)
    probe_all = getattr(provider, "probe_all_datasets_graph_backends", None)
    operation_type = getattr(provider, "AnalysisProviderOperation", None)
    if not callable(probe_all) or operation_type is None:
        return TacticianHammerCapability(
            capability_id,
            TacticianHammerCapabilityStatus.PARTIAL if path else TacticianHammerCapabilityStatus.UNAVAILABLE,
            module_paths=(path,) if path else (),
            producer_id="ipfs-datasets-analysis-provider@1",
            operations=("graph_retrieval", "kg_nomination"),
            diagnostic=_diagnostic(
                TacticianHammerDiagnosticCode.REQUIRED_SYMBOL_MISSING,
                capability_id,
                "analysis provider lacks graph backend probe or operation enum",
                module=provider_module,
            ),
            details={"package_present": bool(path)},
        )
    graph_op = getattr(operation_type, "GRAPH_RETRIEVAL", None)
    graph_value = str(getattr(graph_op, "value", graph_op) or "")
    if graph_value != "graph_retrieval" or not path:
        return TacticianHammerCapability(
            capability_id,
            TacticianHammerCapabilityStatus.PARTIAL if path else TacticianHammerCapabilityStatus.UNAVAILABLE,
            module_paths=(path,) if path else (),
            producer_id="ipfs-datasets-analysis-provider@1",
            operations=("graph_retrieval", "kg_nomination"),
            diagnostic=_diagnostic(
                TacticianHammerDiagnosticCode.PARTIAL_INTERFACE,
                capability_id,
                "GRAPH_RETRIEVAL operation is not bound",
                module=provider_module,
            ),
            details={"package_present": bool(path)},
        )
    return TacticianHammerCapability(
        capability_id,
        TacticianHammerCapabilityStatus.AVAILABLE,
        module_paths=(path,),
        producer_id="ipfs-datasets-analysis-provider@1",
        operations=("graph_retrieval", "kg_nomination", "premise_selection"),
        supported_semantics=(
            "knowledge_graph",
            "graphrag",
            "graph_non_authoritative",
            "nomination_only",
        ),
        details={"operation": graph_value, "semantic_authority": False},
    )


def _run_version(
    capability_id: str,
    command: tuple[str, ...],
    *,
    expected_version: str = "",
    which: Callable[[str], str | None],
    runner: Callable[..., Any],
    timeout_seconds: float,
    operations: tuple[str, ...] = (),
    semantics: tuple[str, ...] = ("version_checked",),
    reconstruction_compatible: bool = False,
) -> TacticianHammerCapability:
    executable = which(command[0])
    if not executable:
        return TacticianHammerCapability(
            capability_id,
            TacticianHammerCapabilityStatus.UNAVAILABLE,
            producer_id=f"toolchain.{command[0]}",
            operations=operations or (command[0],),
            details={"executable_path": "", "version": ""},
            diagnostic=_diagnostic(
                TacticianHammerDiagnosticCode.EXECUTABLE_NOT_FOUND,
                capability_id,
                f"{command[0]} is not on PATH",
            ),
        )
    try:
        completed = runner(
            (executable, *command[1:]),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return TacticianHammerCapability(
            capability_id,
            TacticianHammerCapabilityStatus.TIMED_OUT,
            producer_id=f"toolchain.{command[0]}",
            operations=operations or (command[0],),
            details={"executable_path": executable, "version": ""},
            diagnostic=_diagnostic(
                TacticianHammerDiagnosticCode.PROBE_TIMED_OUT,
                capability_id,
                "version command exceeded probe timeout",
                exception=exc,
            ),
        )
    except OSError as exc:
        return TacticianHammerCapability(
            capability_id,
            TacticianHammerCapabilityStatus.UNAVAILABLE,
            producer_id=f"toolchain.{command[0]}",
            operations=operations or (command[0],),
            details={"executable_path": executable, "version": ""},
            diagnostic=_diagnostic(
                TacticianHammerDiagnosticCode.EXECUTABLE_VERSION_FAILED,
                capability_id,
                "version command could not run",
                exception=exc,
            ),
        )
    output = (
        (getattr(completed, "stdout", "") or "")
        + "\n"
        + (getattr(completed, "stderr", "") or "")
    ).strip()
    if getattr(completed, "returncode", 1) != 0 or not output:
        return TacticianHammerCapability(
            capability_id,
            TacticianHammerCapabilityStatus.UNAVAILABLE,
            producer_id=f"toolchain.{command[0]}",
            operations=operations or (command[0],),
            details={"executable_path": executable, "version": ""},
            diagnostic=_diagnostic(
                TacticianHammerDiagnosticCode.EXECUTABLE_VERSION_FAILED,
                capability_id,
                "version command did not produce a successful version",
                module=executable,
            ),
        )
    version = _first_version(output)
    if expected_version and version != expected_version:
        return TacticianHammerCapability(
            capability_id,
            TacticianHammerCapabilityStatus.INCOMPATIBLE,
            producer_id=f"toolchain.{command[0]}",
            operations=operations or (command[0],),
            details={
                "executable_path": executable,
                "version": version,
                "expected_version": expected_version,
            },
            diagnostic=_diagnostic(
                TacticianHammerDiagnosticCode.EXECUTABLE_VERSION_INCOMPATIBLE,
                capability_id,
                f"expected {expected_version}, got {version or 'unparseable'}",
                module=executable,
            ),
        )
    return TacticianHammerCapability(
        capability_id,
        TacticianHammerCapabilityStatus.AVAILABLE,
        interface_version=version,
        producer_id=f"toolchain.{command[0]}@{version or 'unknown'}",
        operations=operations or (command[0], "version_check"),
        supported_semantics=semantics,
        reconstruction_compatible=reconstruction_compatible,
        details={
            "executable_path": executable,
            "version_output": output,
            # Path/version discovery is not signed supply-chain integrity.
            "signed_binary_integrity": False,
        },
    )


def _first_version(value: str) -> str:
    for token in value.replace("\n", " ").split():
        normalized = token.strip("vV,;()[]")
        if (
            normalized
            and normalized[0].isdigit()
            and all(part.isdigit() for part in normalized.split(".") if part)
        ):
            return normalized
    return ""


def _gitlink_revision(
    root: Path, runner: Callable[..., Any], timeout_seconds: float
) -> tuple[str, TacticianHammerCapabilityDiagnostic | None]:
    try:
        completed = runner(
            ("git", "-C", str(root), "ls-tree", "HEAD", "--", "ipfs_datasets_py"),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        code = (
            TacticianHammerDiagnosticCode.PROBE_TIMED_OUT
            if isinstance(exc, subprocess.TimeoutExpired)
            else TacticianHammerDiagnosticCode.GITLINK_UNAVAILABLE
        )
        return "", _diagnostic(
            code, "datasets.gitlink", "could not read datasets gitlink", exception=exc
        )
    output = (getattr(completed, "stdout", "") or "").strip()
    fields = output.split()
    if (
        getattr(completed, "returncode", 1) != 0
        or len(fields) < 3
        or fields[0] != "160000"
        or len(fields[2]) != 40
    ):
        return "", _diagnostic(
            TacticianHammerDiagnosticCode.GITLINK_MALFORMED,
            "datasets.gitlink",
            "datasets gitlink is missing or malformed",
        )
    return fields[2], None


def _probe_resource_enforcement() -> ResourceEnforcementReport:
    """Type platform CPU/memory enforcement strength without claiming OS isolation."""

    system = platform.system() or "unknown"
    cpu = ResourceEnforcementStrength.UNSUPPORTED
    memory = ResourceEnforcementStrength.UNSUPPORTED
    process = ResourceEnforcementStrength.UNSUPPORTED
    details: dict[str, Any] = {
        "python_implementation": sys.implementation.name,
        "platform_system": system,
        "resource_module": False,
        "rlimit_cpu": False,
        "rlimit_as": False,
        "setsid": hasattr(os, "setsid"),
    }
    try:
        import resource as resource_mod

        details["resource_module"] = True
        details["rlimit_cpu"] = hasattr(resource_mod, "RLIMIT_CPU")
        details["rlimit_as"] = hasattr(resource_mod, "RLIMIT_AS")
        if details["rlimit_cpu"] and details["rlimit_as"] and system != "Windows":
            cpu = ResourceEnforcementStrength.POSIX_RLIMIT
            memory = ResourceEnforcementStrength.POSIX_RLIMIT
            process = (
                ResourceEnforcementStrength.POSIX_RLIMIT
                if details["setsid"]
                else ResourceEnforcementStrength.PARTIAL
            )
        elif details["rlimit_cpu"] or details["rlimit_as"]:
            cpu = (
                ResourceEnforcementStrength.POSIX_RLIMIT
                if details["rlimit_cpu"]
                else ResourceEnforcementStrength.UNSUPPORTED
            )
            memory = (
                ResourceEnforcementStrength.POSIX_RLIMIT
                if details["rlimit_as"]
                else ResourceEnforcementStrength.UNSUPPORTED
            )
            process = ResourceEnforcementStrength.PARTIAL
        else:
            cpu = ResourceEnforcementStrength.UNSUPPORTED
            memory = ResourceEnforcementStrength.UNSUPPORTED
            process = ResourceEnforcementStrength.UNSUPPORTED
    except ImportError:
        details["resource_module"] = False

    return ResourceEnforcementReport(
        platform=system,
        cpu_enforcement=cpu,
        memory_enforcement=memory,
        process_isolation=process,
        # Policy denial is metadata, not an OS isolation receipt.
        network_policy_denied=True,
        network_os_isolation=False,
        environment_lock_path_version_only=True,
        signed_binary_integrity=False,
        details=details,
    )


def _toolchain_capabilities(
    *,
    which: Callable[[str], str | None],
    runner: Callable[..., Any],
    timeout_seconds: float,
) -> tuple[TacticianHammerCapability, ...]:
    return (
        TacticianHammerCapability(
            "toolchain.python",
            TacticianHammerCapabilityStatus.AVAILABLE,
            interface_version=".".join(map(str, sys.version_info[:3])),
            producer_id=f"python@{'.'.join(map(str, sys.version_info[:3]))}",
            operations=("python_runtime",),
            supported_semantics=("python_runtime", "host_interpreter"),
            details={
                "executable_path": sys.executable,
                "implementation": sys.implementation.name,
                "signed_binary_integrity": False,
            },
        ),
        _run_version(
            "toolchain.z3",
            ("z3", "--version"),
            which=which,
            runner=runner,
            timeout_seconds=timeout_seconds,
            operations=("smt_solve", "z3"),
            semantics=("z3", "solver_candidates_non_authoritative", "version_checked"),
        ),
        _run_version(
            "toolchain.cvc5",
            ("cvc5", "--version"),
            expected_version=PINNED_CVC5_VERSION,
            which=which,
            runner=runner,
            timeout_seconds=timeout_seconds,
            operations=("smt_solve", "cvc5"),
            semantics=("cvc5", "solver_candidates_non_authoritative", "version_checked"),
        ),
        _run_version(
            "toolchain.vampire",
            ("vampire", "--version"),
            which=which,
            runner=runner,
            timeout_seconds=timeout_seconds,
            operations=("atp_solve", "vampire"),
            semantics=(
                "vampire",
                "solver_candidates_non_authoritative",
                "version_checked",
            ),
        ),
        # E's CLI is historically `eprover`; capability id remains toolchain.e.
        _run_version(
            "toolchain.e",
            ("eprover", "--version"),
            which=which,
            runner=runner,
            timeout_seconds=timeout_seconds,
            operations=("atp_solve", "e"),
            semantics=("e", "solver_candidates_non_authoritative", "version_checked"),
        ),
        _run_version(
            "toolchain.lean",
            ("lean", "--version"),
            which=which,
            runner=runner,
            timeout_seconds=timeout_seconds,
            operations=("lean_kernel", "itp"),
            semantics=("lean", "kernel_reconstruction", "version_checked"),
            reconstruction_compatible=True,
        ),
        _run_version(
            "toolchain.coq",
            ("coqc", "--version"),
            which=which,
            runner=runner,
            timeout_seconds=timeout_seconds,
            operations=("coq_kernel", "itp"),
            semantics=("coq", "kernel_reconstruction", "version_checked"),
            reconstruction_compatible=True,
        ),
        _run_version(
            "toolchain.isabelle",
            ("isabelle", "version"),
            which=which,
            runner=runner,
            timeout_seconds=timeout_seconds,
            operations=("isabelle_kernel", "itp"),
            semantics=("isabelle", "kernel_reconstruction", "version_checked"),
            reconstruction_compatible=True,
        ),
        _run_version(
            "toolchain.mypy",
            ("mypy", "--version"),
            which=which,
            runner=runner,
            timeout_seconds=timeout_seconds,
            operations=("mypy_typecheck",),
            semantics=("mypy", "static_types", "version_checked"),
        ),
    )


def probe_tactician_hammer_capabilities(
    *,
    importer: Callable[[str], Any] | None = None,
    which: Callable[[str], str | None] | None = None,
    runner: Callable[..., Any] | None = None,
    timeout_seconds: float = DEFAULT_CAPABILITY_PROBE_TIMEOUT_SECONDS,
    repository_root: Path | str | None = None,
    learned_selector_admitted: bool = False,
    model_execution_admitted: bool = False,
    native_execution_admitted: bool = False,
    network_access_admitted: bool = False,
    auto_install_admitted: bool = False,
) -> TacticianHammerCapabilityReport:
    """Probe exact Tactician, Hammer, and static-analysis capabilities.

    Injection points make failures, partial interfaces, and timeouts testable
    without mutating process-wide import state.  A timeout is a diagnosis, not
    an optimistic availability result.  Package presence and solver/model
    candidates grant no authority.

    Learned/model/native/network/install features remain off unless the
    corresponding ``*_admitted`` flag is explicitly true.  This probe never
    sets those flags itself and never installs, networks, or executes native
    proof search.
    """

    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds must be a positive number")
    for name, value in (
        ("learned_selector_admitted", learned_selector_admitted),
        ("model_execution_admitted", model_execution_admitted),
        ("native_execution_admitted", native_execution_admitted),
        ("network_access_admitted", network_access_admitted),
        ("auto_install_admitted", auto_install_admitted),
    ):
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be a boolean")

    load = importer or importlib.import_module
    locate = which or find_contract_repair_executable
    execute = runner or subprocess.run
    started = time.monotonic()
    root = (
        Path(repository_root)
        if repository_root is not None
        else Path(__file__).resolve().parents[3]
    )

    capabilities: list[TacticianHammerCapability] = []

    # Domain-neutral Tactician (pending LPR-003 when absent) and legal adapter.
    capabilities.append(
        _probe_generic_tactician(importer=load, timeout_seconds=float(timeout_seconds))
    )
    capabilities.append(
        _probe_legal_tactician_adapter(
            importer=load, timeout_seconds=float(timeout_seconds)
        )
    )

    # Import isolation of the production Hammer lazy-load path.
    isolation_cap, isolation_state = _probe_import_isolation(
        importer=load, timeout_seconds=float(timeout_seconds)
    )
    capabilities.append(isolation_cap)

    # Exact Hammer descriptors (corpus, selectors, translation, reconstruction,
    # receipts).  These import descriptor modules only — never the unsafe
    # process-global load path and never proof search.
    for spec in _HAMMER_SPECS:
        cap = _probe_interface(spec, importer=load, timeout_seconds=float(timeout_seconds))
        # Learned selector descriptor may be present, but feature stays off
        # unless explicitly admitted.
        if (
            spec.capability_id == "hammer.selector.learned"
            and cap.available
            and not learned_selector_admitted
        ):
            details = dict(cap.details)
            details["feature_admitted"] = False
            details["admission_required"] = "learned_selector_admitted"
            capabilities.append(
                TacticianHammerCapability(
                    cap.capability_id,
                    TacticianHammerCapabilityStatus.AVAILABLE,
                    module_paths=cap.module_paths,
                    interface_version=cap.interface_version,
                    schema_version=cap.schema_version,
                    producer_id=cap.producer_id,
                    operations=cap.operations,
                    supported_semantics=cap.supported_semantics
                    + ("feature_not_admitted_by_default",),
                    details=details,
                )
            )
        else:
            capabilities.append(cap)

    # Static analysis, vector, KG, llm_router, and logic provider interfaces.
    for spec in _STATIC_AND_AUX_SPECS:
        capabilities.append(
            _probe_interface(spec, importer=load, timeout_seconds=float(timeout_seconds))
        )
    capabilities.append(
        _probe_kg_provider(importer=load, timeout_seconds=float(timeout_seconds))
    )

    # Solver and ITP executables: version-checked only; never proof search.
    capabilities.extend(
        _toolchain_capabilities(
            which=locate, runner=execute, timeout_seconds=float(timeout_seconds)
        )
    )

    gitlink_revision, gitlink_diagnostic = _gitlink_revision(
        root, execute, float(timeout_seconds)
    )
    if gitlink_diagnostic:
        capabilities.append(
            TacticianHammerCapability(
                "datasets.gitlink",
                TacticianHammerCapabilityStatus.UNAVAILABLE,
                producer_id="datasets-gitlink@1",
                operations=("gitlink_revision",),
                diagnostic=gitlink_diagnostic,
            )
        )
    else:
        capabilities.append(
            TacticianHammerCapability(
                "datasets.gitlink",
                TacticianHammerCapabilityStatus.AVAILABLE,
                producer_id="datasets-gitlink@1",
                operations=("gitlink_revision",),
                supported_semantics=("gitlink_revision_bound",),
                details={"executable_path": "git", "revision": gitlink_revision},
            )
        )

    # Explicit non-admission records so callers cannot confuse defaults with
    # silent enablement.
    if not native_execution_admitted:
        capabilities.append(
            TacticianHammerCapability(
                "feature.native_execution",
                TacticianHammerCapabilityStatus.UNAVAILABLE,
                producer_id="feature-admission@1",
                operations=("native_solver", "native_kernel"),
                diagnostic=_diagnostic(
                    TacticianHammerDiagnosticCode.FEATURE_NOT_ADMITTED,
                    "feature.native_execution",
                    "native solver/frontend/kernel execution remains off unless "
                    "explicitly admitted",
                ),
                details={"feature_admitted": False},
            )
        )
    else:
        capabilities.append(
            TacticianHammerCapability(
                "feature.native_execution",
                TacticianHammerCapabilityStatus.AVAILABLE,
                producer_id="feature-admission@1",
                operations=("native_solver", "native_kernel"),
                supported_semantics=("explicitly_admitted",),
                details={
                    "feature_admitted": True,
                    "executable_path": "admitted",
                },
            )
        )
    if not network_access_admitted:
        capabilities.append(
            TacticianHammerCapability(
                "feature.network",
                TacticianHammerCapabilityStatus.UNAVAILABLE,
                producer_id="feature-admission@1",
                operations=("network_access",),
                diagnostic=_diagnostic(
                    TacticianHammerDiagnosticCode.FEATURE_NOT_ADMITTED,
                    "feature.network",
                    "network remains denied by default; policy denial is not OS isolation",
                ),
                details={
                    "feature_admitted": False,
                    "network_policy_denied": True,
                    "network_os_isolation": False,
                },
            )
        )
    else:
        capabilities.append(
            TacticianHammerCapability(
                "feature.network",
                TacticianHammerCapabilityStatus.AVAILABLE,
                producer_id="feature-admission@1",
                operations=("network_access",),
                supported_semantics=("explicitly_admitted", "not_os_isolation"),
                details={
                    "feature_admitted": True,
                    "network_os_isolation": False,
                    "executable_path": "admitted",
                },
            )
        )
    if not auto_install_admitted:
        capabilities.append(
            TacticianHammerCapability(
                "feature.auto_install",
                TacticianHammerCapabilityStatus.UNAVAILABLE,
                producer_id="feature-admission@1",
                operations=("install",),
                diagnostic=_diagnostic(
                    TacticianHammerDiagnosticCode.FEATURE_NOT_ADMITTED,
                    "feature.auto_install",
                    "auto-install remains off unless explicitly admitted",
                ),
                details={"feature_admitted": False},
            )
        )
    else:
        capabilities.append(
            TacticianHammerCapability(
                "feature.auto_install",
                TacticianHammerCapabilityStatus.AVAILABLE,
                producer_id="feature-admission@1",
                operations=("install",),
                supported_semantics=("explicitly_admitted",),
                details={"feature_admitted": True, "executable_path": "admitted"},
            )
        )
    if not model_execution_admitted:
        capabilities.append(
            TacticianHammerCapability(
                "feature.model_execution",
                TacticianHammerCapabilityStatus.UNAVAILABLE,
                producer_id="feature-admission@1",
                operations=("model_execution",),
                diagnostic=_diagnostic(
                    TacticianHammerDiagnosticCode.FEATURE_NOT_ADMITTED,
                    "feature.model_execution",
                    "model/LLM execution remains off unless explicitly admitted",
                ),
                details={"feature_admitted": False},
            )
        )
    else:
        capabilities.append(
            TacticianHammerCapability(
                "feature.model_execution",
                TacticianHammerCapabilityStatus.AVAILABLE,
                producer_id="feature-admission@1",
                operations=("model_execution",),
                supported_semantics=("explicitly_admitted", "proposal_only"),
                details={"feature_admitted": True, "executable_path": "admitted"},
            )
        )

    resource_enforcement = _probe_resource_enforcement()
    all_diagnostics = tuple(
        item.diagnostic for item in capabilities if item.diagnostic is not None
    )
    datasets_paths = tuple(
        sorted(
            {
                path
                for item in capabilities
                if item.capability_id.startswith(
                    ("datasets.", "hammer.", "tactician.", "kg.")
                )
                for path in item.module_paths
            }
        )
    )
    accelerator_prefixes = (
        "analyzer.",
        "vector.",
        "llm.",
        "datasets.logic_provider",
        "datasets.analysis_provider",
        "hammer.import_isolation",
    )
    accelerator_paths = tuple(
        sorted(
            {
                path
                for item in capabilities
                if item.capability_id.startswith(accelerator_prefixes)
                for path in item.module_paths
            }
            | {os.path.realpath(__file__)}
        )
    )
    return TacticianHammerCapabilityReport(
        capabilities=tuple(sorted(capabilities, key=lambda item: item.capability_id)),
        accelerator_module_paths=accelerator_paths,
        datasets_module_paths=datasets_paths,
        datasets_gitlink_revision=gitlink_revision,
        import_isolation=isolation_state,
        resource_enforcement=resource_enforcement,
        diagnostics=all_diagnostics,
        generated_at_monotonic=started,
        duration_seconds=time.monotonic() - started,
        learned_selector_admitted=learned_selector_admitted,
        model_execution_admitted=model_execution_admitted,
        native_execution_admitted=native_execution_admitted,
        network_access_admitted=network_access_admitted,
        auto_install_admitted=auto_install_admitted,
    )


__all__ = [
    "DEFAULT_CAPABILITY_PROBE_TIMEOUT_SECONDS",
    "GENERIC_TACTICIAN_MODULE",
    "IMPORT_ISOLATION_HARDENED",
    "IMPORT_ISOLATION_UNSAFE",
    "PINNED_CVC5_VERSION",
    "ResourceEnforcementReport",
    "ResourceEnforcementStrength",
    "TACTICIAN_HAMMER_CAPABILITY_REPORT_SCHEMA_VERSION",
    "TACTICIAN_HAMMER_CAPABILITY_REPORT_VERSION",
    "TACTICIAN_HAMMER_CAPABILITY_SCHEMA_VERSION",
    "TacticianHammerCapability",
    "TacticianHammerCapabilityDiagnostic",
    "TacticianHammerCapabilityReport",
    "TacticianHammerCapabilityStatus",
    "TacticianHammerDiagnosticCode",
    "probe_tactician_hammer_capabilities",
]
