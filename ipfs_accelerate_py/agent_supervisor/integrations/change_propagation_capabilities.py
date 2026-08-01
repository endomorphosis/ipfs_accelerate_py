"""Lazy, fail-closed capability admission for change-propagation work.

Importing this module is deliberately cheap: repository/AST indexes, program
graph façades, impact/dataflow graphs, vector nomination, datasets GraphRAG
and logic backends, solver executables, language toolchains, and the canonical
``llm_router`` / provider-receipt surfaces are inspected only by
``probe_change_propagation_capabilities``.

Discovery is not authority.  Package presence, solver binaries, model
candidates, GraphRAG hits, and vector nearest-neighbours never promote a
capability to semantic, proof, or completion authority.  Missing, partial,
incompatible, or timed-out surfaces yield typed unavailable diagnostics.

The probe never installs packages, invokes package managers, or contacts a
network service.
"""

from __future__ import annotations

import importlib
import os
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
    PINNED_TYPESCRIPT_VERSION,
    find_contract_repair_executable,
)

CHANGE_PROPAGATION_CAPABILITY_SCHEMA_VERSION: Final = (
    "ipfs_accelerate_py/agent-supervisor/change-propagation-capability@1"
)
CHANGE_PROPAGATION_CAPABILITY_REPORT_SCHEMA_VERSION: Final = (
    "ipfs_accelerate_py/agent-supervisor/change-propagation-capability-report@1"
)
CHANGE_PROPAGATION_CAPABILITY_REPORT_VERSION: Final = 1
# Cold imports from the exact datasets gitlink can legitimately take several
# seconds.  Two seconds caused false timeouts for healthy logic backends.
DEFAULT_CAPABILITY_PROBE_TIMEOUT_SECONDS: Final = 10.0


class ChangePropagationCapabilityStatus(str, Enum):
    """Closed admission outcomes; only ``AVAILABLE`` admits an interface."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    UNSUPPORTED = "unsupported"
    INCOMPATIBLE = "incompatible"
    PARTIAL = "partial"
    TIMED_OUT = "timed_out"


class ChangePropagationDiagnosticCode(str, Enum):
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
    INTERNAL_ERROR = "internal_error"


@dataclass(frozen=True)
class ChangePropagationCapabilityDiagnostic:
    """Typed, bounded probe diagnostic rather than an exception-only failure."""

    code: ChangePropagationDiagnosticCode
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
class ChangePropagationCapability:
    """One exact capability binding used by change-propagation admission."""

    capability_id: str
    status: ChangePropagationCapabilityStatus
    module_paths: tuple[str, ...] = ()
    interface_version: str = ""
    schema_version: str = ""
    producer_id: str = ""
    operations: tuple[str, ...] = ()
    supported_semantics: tuple[str, ...] = ()
    diagnostic: ChangePropagationCapabilityDiagnostic | None = None
    reconstruction_compatible: bool = False
    candidate_authoritative: bool = False
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.capability_id.strip():
            raise ValueError("capability_id must not be empty")
        if self.status is ChangePropagationCapabilityStatus.AVAILABLE:
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
        return self.status is ChangePropagationCapabilityStatus.AVAILABLE

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
class ChangePropagationCapabilityReport:
    """Immutable, versioned snapshot of change-propagation prerequisites."""

    capabilities: tuple[ChangePropagationCapability, ...]
    accelerator_module_paths: tuple[str, ...]
    datasets_module_paths: tuple[str, ...]
    datasets_gitlink_revision: str
    diagnostics: tuple[ChangePropagationCapabilityDiagnostic, ...] = ()
    generated_at_monotonic: float = 0.0
    duration_seconds: float = 0.0
    schema_version: str = CHANGE_PROPAGATION_CAPABILITY_REPORT_SCHEMA_VERSION
    report_version: int = CHANGE_PROPAGATION_CAPABILITY_REPORT_VERSION

    def __post_init__(self) -> None:
        ids = [item.capability_id for item in self.capabilities]
        if len(ids) != len(set(ids)):
            raise ValueError("capability ids must be unique")
        if self.report_version != CHANGE_PROPAGATION_CAPABILITY_REPORT_VERSION:
            raise ValueError("unsupported capability report version")
        if self.duration_seconds < 0:
            raise ValueError("duration_seconds must be non-negative")
        object.__setattr__(self, "capabilities", tuple(self.capabilities))
        object.__setattr__(
            self,
            "accelerator_module_paths",
            tuple(sorted(set(self.accelerator_module_paths))),
        )
        object.__setattr__(
            self, "datasets_module_paths", tuple(sorted(set(self.datasets_module_paths)))
        )
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))

    @property
    def capability_map(self) -> Mapping[str, ChangePropagationCapability]:
        return MappingProxyType({item.capability_id: item for item in self.capabilities})

    @property
    def capabilities_by_id(self) -> Mapping[str, ChangePropagationCapability]:
        return self.capability_map

    def capability(self, capability_id: str) -> ChangePropagationCapability:
        try:
            return self.capability_map[capability_id]
        except KeyError as exc:
            raise KeyError(
                f"unknown change-propagation capability: {capability_id}"
            ) from exc

    @property
    def toolchains(self) -> Mapping[str, ChangePropagationCapability]:
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

    @property
    def interface_versions(self) -> Mapping[str, str]:
        return MappingProxyType(
            {item.capability_id: item.interface_version for item in self.capabilities}
        )

    @property
    def schema_versions(self) -> Mapping[str, str]:
        return MappingProxyType(
            {item.capability_id: item.schema_version for item in self.capabilities}
        )

    @property
    def operations(self) -> Mapping[str, tuple[str, ...]]:
        return MappingProxyType(
            {item.capability_id: item.operations for item in self.capabilities}
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "report_version": self.report_version,
            "duration_seconds": self.duration_seconds,
            "accelerator_module_paths": list(self.accelerator_module_paths),
            "datasets_module_paths": list(self.datasets_module_paths),
            "datasets_gitlink_revision": self.datasets_gitlink_revision,
            "capabilities": {
                item.capability_id: item.to_dict() for item in self.capabilities
            },
            "toolchains": {
                item.capability_id: item.to_dict() for item in self.toolchains.values()
            },
            "diagnostics": [item.to_dict() for item in self.diagnostics],
            "network_access": False,
            "auto_install": False,
            "solver_candidates_authoritative": False,
            "vector_semantic_authority": False,
            "graph_semantic_authority": False,
            "llm_completion_authority": False,
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


# Exact local interfaces for change-propagation stages.  Modules that do not
# yet exist (program_graph / program_call_resolver façades) correctly report
# typed UNAVAILABLE until RPR-025 lands them; package markers never fill gaps.
_INTERFACE_SPECS: Final = (
    _InterfaceSpec(
        "index.repository",
        "ipfs_accelerate_py.agent_supervisor.analysis.repository_indexer",
        ("RepositoryIndexer", "REPOSITORY_INDEX_SCHEMA"),
        interface_constant="REPOSITORY_INDEXER_VERSION",
        schema_constant="REPOSITORY_INDEX_SCHEMA",
        producer_id="repository-indexer@1",
        operations=("repository_index", "multi_root_index"),
        semantics=("repository_index", "snapshot_bound", "compact_rows"),
    ),
    _InterfaceSpec(
        "index.analysis_ast",
        "ipfs_accelerate_py.agent_supervisor.analysis.analysis_ast_index",
        ("AnalysisASTIndex", "build_analysis_ast_index"),
        schema_constant="ANALYSIS_AST_INDEX_SCHEMA",
        producer_id="analysis-ast-index@1",
        operations=("ast_index", "ast_query"),
        semantics=("analysis_ast_index", "snapshot_bound"),
    ),
    _InterfaceSpec(
        "graph.program_graph",
        "ipfs_accelerate_py.agent_supervisor.program_graph",
        ("ProgramGraph",),
        producer_id="program-graph@1",
        operations=("trace_graph_evidence", "call_edges", "data_edges"),
        semantics=("program_graph", "complete_frontier", "unknown_frontier"),
    ),
    _InterfaceSpec(
        "graph.program_call_resolver",
        "ipfs_accelerate_py.agent_supervisor.program_call_resolver",
        ("ProgramCallResolver",),
        producer_id="program-call-resolver@1",
        operations=("resolve_call",),
        semantics=("conservative_call_resolution", "unknown_frontier"),
    ),
    _InterfaceSpec(
        "graph.code_impact",
        "ipfs_accelerate_py.agent_supervisor.analysis.code_evidence_graph",
        ("CodeImpactIndex", "CodeEvidenceGraph"),
        schema_constant="CODE_IMPACT_INDEX_SCHEMA",
        producer_id="code-impact-index@1",
        operations=("impact_selection", "reverse_closure"),
        semantics=("impact_graph", "tree_bound", "dependent_to_providers"),
    ),
    _InterfaceSpec(
        "graph.semantic_dependency",
        "ipfs_accelerate_py.agent_supervisor.analysis.semantic_dependency_graph",
        (
            "SemanticDependencyGraph",
            "build_semantic_dependency_graph",
            "compute_mandatory_closure",
        ),
        schema_constant="SEMANTIC_DEPENDENCY_GRAPH_SCHEMA",
        producer_id="semantic-dependency-graph@1",
        operations=("build_graph", "mandatory_closure"),
        semantics=(
            "semantic_dependency_graph",
            "non_authoritative_nominated_edges",
            "mandatory_closure",
        ),
    ),
    _InterfaceSpec(
        "graph.value_provenance",
        "ipfs_accelerate_py.agent_supervisor.analysis.code_evidence_graph",
        ("ProvenanceEdge", "EvidenceProvenance", "CodeEvidenceGraph"),
        schema_constant="CODE_EVIDENCE_EDGE_SCHEMA",
        producer_id="code-evidence-graph@1",
        operations=("provenance_edge", "value_flow"),
        semantics=("value_provenance", "provenance_bearing_edges"),
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
        "datasets.analysis_provider",
        "ipfs_accelerate_py.agent_supervisor.integrations.ipfs_datasets_analysis_provider",
        (
            "IpfsDatasetsAnalysisProvider",
            "AnalysisProviderOperation",
            "probe_all_datasets_graph_backends",
        ),
        producer_id="ipfs-datasets-analysis-provider@1",
        operations=(
            "graph_retrieval",
            "premise_selection",
            "provenance_query",
            "symbol_impact",
        ),
        semantics=(
            "IPFSDatasetsAnalysisProvider",
            "GraphRAG_non_authoritative",
            "premise_selection",
        ),
    ),
    _InterfaceSpec(
        "datasets.logic_provider",
        "ipfs_accelerate_py.agent_supervisor.integrations.ipfs_datasets_logic_provider",
        (
            "IpfsDatasetsLogicProvider",
            "DatasetsLogicBackendProbe",
            "probe_all_datasets_logic_backends",
        ),
        interface_constant="IPFS_DATASETS_LOGIC_PROVIDER_VERSION",
        schema_constant="HAMMER_ADAPTER_SCHEMA_VERSION",
        producer_id="ipfs-datasets-logic-provider@1",
        operations=(
            "logic_ir",
            "tdfol",
            "cec",
            "smt",
            "hammer",
            "reconstruction",
        ),
        semantics=(
            "IPFSDatasetsLogicProvider",
            "BackendCapability",
            "solver_candidates_non_authoritative",
            "independent_reconstruction_required",
        ),
        reconstruction_compatible=True,
    ),
    _InterfaceSpec(
        "llm.router",
        "ipfs_accelerate_py.llm_router",
        (
            "generate_text",
            "get_last_usage_admission",
            "get_last_generation_trace",
        ),
        producer_id="llm-router@1",
        operations=("text.generate", "text.chat"),
        semantics=(
            "llm_router",
            "proposal_only",
            "non_authoritative",
            "no_completion_authority",
        ),
    ),
    _InterfaceSpec(
        "llm.provider_receipt",
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.contract_packet_provider_router",
        ("ProviderExecutionReceipt",),
        interface_constant="PROVIDER_EXECUTION_RECEIPT_INTERFACE",
        expected_interface="ProviderExecutionReceipt@1",
        schema_constant="PROVIDER_EXECUTION_RECEIPT_SCHEMA",
        producer_id="implementation-provider-router@1",
        operations=("provider_execution_receipt", "route_proposal"),
        semantics=(
            "ProviderExecutionReceipt@1",
            "provider_output_tier_proposal",
            "no_repository_write_authority",
            "no_proof_authority",
            "no_completion_authority",
        ),
    ),
)


def _diagnostic(
    code: ChangePropagationDiagnosticCode,
    capability_id: str,
    message: str,
    *,
    module: str = "",
    exception: BaseException | None = None,
) -> ChangePropagationCapabilityDiagnostic:
    return ChangePropagationCapabilityDiagnostic(
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
) -> ChangePropagationCapability:
    completed, module, error = _bounded_call(
        lambda: importer(spec.module), timeout_seconds
    )
    if not completed:
        return ChangePropagationCapability(
            spec.capability_id,
            ChangePropagationCapabilityStatus.TIMED_OUT,
            producer_id=spec.producer_id,
            operations=spec.operations,
            diagnostic=_diagnostic(
                ChangePropagationDiagnosticCode.PROBE_TIMED_OUT,
                spec.capability_id,
                "module import exceeded probe timeout",
                module=spec.module,
            ),
        )
    if error is not None:
        return ChangePropagationCapability(
            spec.capability_id,
            ChangePropagationCapabilityStatus.UNAVAILABLE,
            producer_id=spec.producer_id,
            operations=spec.operations,
            diagnostic=_diagnostic(
                ChangePropagationDiagnosticCode.MODULE_IMPORT_FAILED,
                spec.capability_id,
                "required module could not be imported",
                module=spec.module,
                exception=error,
            ),
        )
    path = _module_path(module)
    if not path:
        return ChangePropagationCapability(
            spec.capability_id,
            ChangePropagationCapabilityStatus.INCOMPATIBLE,
            producer_id=spec.producer_id,
            operations=spec.operations,
            diagnostic=_diagnostic(
                ChangePropagationDiagnosticCode.MODULE_PATH_UNAVAILABLE,
                spec.capability_id,
                "imported module has no exact file path",
                module=spec.module,
            ),
        )
    missing = [symbol for symbol in spec.symbols if not hasattr(module, symbol)]
    if missing:
        return ChangePropagationCapability(
            spec.capability_id,
            ChangePropagationCapabilityStatus.PARTIAL,
            module_paths=(path,),
            producer_id=spec.producer_id,
            operations=spec.operations,
            diagnostic=_diagnostic(
                ChangePropagationDiagnosticCode.REQUIRED_SYMBOL_MISSING,
                spec.capability_id,
                f"missing required interface symbols: {', '.join(missing)}",
                module=spec.module,
            ),
        )
    interface_version = (
        str(getattr(module, spec.interface_constant, ""))
        if spec.interface_constant
        else ""
    )
    if spec.expected_interface and interface_version != spec.expected_interface:
        return ChangePropagationCapability(
            spec.capability_id,
            ChangePropagationCapabilityStatus.INCOMPATIBLE,
            module_paths=(path,),
            interface_version=interface_version,
            producer_id=spec.producer_id,
            operations=spec.operations,
            diagnostic=_diagnostic(
                ChangePropagationDiagnosticCode.INTERFACE_VERSION_INCOMPATIBLE,
                spec.capability_id,
                f"expected interface version {spec.expected_interface!r}, "
                f"got {interface_version!r}",
                module=spec.module,
            ),
        )
    schema_version = (
        str(getattr(module, spec.schema_constant, "")) if spec.schema_constant else ""
    )
    if spec.expected_schema and schema_version != spec.expected_schema:
        return ChangePropagationCapability(
            spec.capability_id,
            ChangePropagationCapabilityStatus.INCOMPATIBLE,
            module_paths=(path,),
            interface_version=interface_version,
            schema_version=schema_version,
            producer_id=spec.producer_id,
            operations=spec.operations,
            diagnostic=_diagnostic(
                ChangePropagationDiagnosticCode.SCHEMA_VERSION_INCOMPATIBLE,
                spec.capability_id,
                f"expected schema version {spec.expected_schema!r}, "
                f"got {schema_version!r}",
                module=spec.module,
            ),
        )
    return ChangePropagationCapability(
        spec.capability_id,
        ChangePropagationCapabilityStatus.AVAILABLE,
        module_paths=(path,),
        interface_version=interface_version,
        schema_version=schema_version,
        producer_id=spec.producer_id,
        operations=spec.operations,
        supported_semantics=spec.semantics,
        reconstruction_compatible=spec.reconstruction_compatible,
        details={"module": spec.module, "symbols": list(spec.symbols)},
    )


def _resolve_symbol_paths(
    module_names: Sequence[str],
    *,
    importer: Callable[[str], Any],
    timeout_seconds: float,
) -> dict[str, str]:
    """Bind physical paths for already-admitted exact module names only."""

    paths_by_module: dict[str, str] = {}
    for module_name in module_names:
        checked_module = sys.modules.get(module_name)
        if checked_module is None:
            completed_path, checked_module, path_error = _bounded_call(
                lambda module_name=module_name: importer(module_name),
                timeout_seconds,
            )
            if not completed_path or path_error is not None:
                continue
        path = _module_path(checked_module)
        if path:
            paths_by_module[module_name] = path
    return paths_by_module


def _probe_datasets_logic_backends(
    *, importer: Callable[[str], Any], timeout_seconds: float
) -> tuple[ChangePropagationCapability, ...]:
    """Adapt the exact-symbol datasets logic probe without trusting labels."""

    provider_module = (
        "ipfs_accelerate_py.agent_supervisor.integrations.ipfs_datasets_logic_provider"
    )
    capability_ids = (
        "datasets.logic_ir",
        "datasets.tdfol",
        "datasets.cec",
        "datasets.smt",
        "datasets.hammer",
        "datasets.reconstruction",
    )
    completed, provider, error = _bounded_call(
        lambda: importer(provider_module), timeout_seconds
    )
    if not completed or error is not None:
        code = (
            ChangePropagationDiagnosticCode.PROBE_TIMED_OUT
            if not completed
            else ChangePropagationDiagnosticCode.MODULE_IMPORT_FAILED
        )
        status = (
            ChangePropagationCapabilityStatus.TIMED_OUT
            if not completed
            else ChangePropagationCapabilityStatus.UNAVAILABLE
        )
        return tuple(
            ChangePropagationCapability(
                item,
                status,
                producer_id="ipfs-datasets-logic-provider@1",
                operations=(item.split(".", 1)[-1],),
                diagnostic=_diagnostic(
                    code,
                    item,
                    "datasets logic probe adapter unavailable",
                    module=provider_module,
                    exception=error,
                ),
            )
            for item in capability_ids
        )
    probe_all = getattr(provider, "probe_all_datasets_logic_backends", None)
    kind_type = getattr(provider, "DatasetsLogicBackendKind", None)
    if not callable(probe_all) or kind_type is None:
        return tuple(
            ChangePropagationCapability(
                item,
                ChangePropagationCapabilityStatus.INCOMPATIBLE,
                producer_id="ipfs-datasets-logic-provider@1",
                operations=(item.split(".", 1)[-1],),
                diagnostic=_diagnostic(
                    ChangePropagationDiagnosticCode.REQUIRED_SYMBOL_MISSING,
                    item,
                    "datasets logic adapter lacks exact backend probe",
                    module=provider_module,
                ),
            )
            for item in capability_ids
        )
    completed, probes, error = _bounded_call(
        lambda: probe_all(importer=importer), timeout_seconds
    )
    if not completed or error is not None:
        code = (
            ChangePropagationDiagnosticCode.PROBE_TIMED_OUT
            if not completed
            else ChangePropagationDiagnosticCode.INTERNAL_ERROR
        )
        status = (
            ChangePropagationCapabilityStatus.TIMED_OUT
            if not completed
            else ChangePropagationCapabilityStatus.UNAVAILABLE
        )
        return tuple(
            ChangePropagationCapability(
                item,
                status,
                producer_id="ipfs-datasets-logic-provider@1",
                operations=(item.split(".", 1)[-1],),
                diagnostic=_diagnostic(
                    code,
                    item,
                    "datasets logic backend probe failed",
                    module=provider_module,
                    exception=error,
                ),
            )
            for item in capability_ids
        )

    result: list[ChangePropagationCapability] = []
    reconstruction_paths: list[str] = []
    reconstruction_semantics: list[str] = []
    any_reconstruction = False
    for probe in probes:
        kind = str(getattr(getattr(probe, "kind", None), "value", ""))
        capability_id = f"datasets.{'logic_ir' if kind == 'ir' else kind}"
        receipts = tuple(getattr(probe, "symbol_receipts", ()))
        module_names = tuple(
            str(getattr(item, "module", ""))
            for item in receipts
            if getattr(item, "available", False)
        )
        paths_by_module = _resolve_symbol_paths(
            module_names, importer=importer, timeout_seconds=timeout_seconds
        )
        paths = tuple(sorted(set(paths_by_module.values())))
        available = (
            bool(getattr(probe, "available", False))
            and bool(receipts)
            and all(bool(getattr(item, "available", False)) for item in receipts)
            and len(paths_by_module) == len(set(module_names))
        )
        reconstruction_compatible = bool(
            getattr(probe, "reconstruction_compatible", False)
        )
        if reconstruction_compatible and available:
            any_reconstruction = True
            reconstruction_paths.extend(paths)
            reconstruction_semantics.append(kind)
        if available:
            result.append(
                ChangePropagationCapability(
                    capability_id,
                    ChangePropagationCapabilityStatus.AVAILABLE,
                    module_paths=paths,
                    interface_version=str(
                        getattr(provider, "LOGIC_IR_INTERFACE", "")
                    ),
                    schema_version=str(
                        getattr(provider, "DATASETS_LOGIC_PROBE_SCHEMA", "")
                    ),
                    producer_id=str(getattr(probe, "provider_id", "") or kind),
                    operations=(kind, "solver_candidate"),
                    supported_semantics=(
                        kind,
                        "solver_candidates_non_authoritative",
                        "independent_reconstruction_required",
                    ),
                    reconstruction_compatible=reconstruction_compatible,
                    details={
                        "capability_revision": str(
                            getattr(probe, "capability_revision", "")
                        ),
                        "package_version": str(
                            getattr(probe, "package_version", "")
                        ),
                        "provider_id": str(getattr(probe, "provider_id", "")),
                    },
                )
            )
        else:
            result.append(
                ChangePropagationCapability(
                    capability_id,
                    (
                        ChangePropagationCapabilityStatus.PARTIAL
                        if paths
                        else ChangePropagationCapabilityStatus.UNAVAILABLE
                    ),
                    module_paths=paths,
                    producer_id=str(getattr(probe, "provider_id", "") or kind),
                    operations=(kind,),
                    diagnostic=_diagnostic(
                        (
                            ChangePropagationDiagnosticCode.PARTIAL_INTERFACE
                            if paths
                            else ChangePropagationDiagnosticCode.MODULE_IMPORT_FAILED
                        ),
                        capability_id,
                        str(
                            getattr(
                                probe,
                                "unavailable_reason",
                                "required symbols unavailable",
                            )
                        ),
                        module=", ".join(module_names),
                    ),
                    reconstruction_compatible=reconstruction_compatible,
                    details={
                        "reason_code": str(getattr(probe, "reason_code", "")),
                    },
                )
            )

    # Reconstruction is an independent authority surface: available only when
    # at least one reconstruction-compatible backend is fully bound.  Solver
    # presence alone never grants it.
    provider_cls = getattr(provider, "IpfsDatasetsLogicProvider", None)
    has_reconstruct_api = provider_cls is not None and (
        callable(getattr(provider_cls, "reconstruct", None))
        or callable(getattr(provider, "reconstruct", None))
    )
    if any_reconstruction and (has_reconstruct_api or provider_cls is not None):
        result.append(
            ChangePropagationCapability(
                "datasets.reconstruction",
                ChangePropagationCapabilityStatus.AVAILABLE,
                module_paths=tuple(sorted(set(reconstruction_paths)))
                or (_module_path(provider),),
                interface_version=str(getattr(provider, "LOGIC_IR_INTERFACE", "")),
                schema_version=str(
                    getattr(provider, "DATASETS_LOGIC_PROBE_SCHEMA", "")
                ),
                producer_id="kernel-reconstruction@1",
                operations=("reconstruction", "independent_kernel_reconstruction"),
                supported_semantics=(
                    "independent_reconstruction_required",
                    "solver_candidates_non_authoritative",
                    *reconstruction_semantics,
                ),
                reconstruction_compatible=True,
                details={
                    "reconstruction_backends": sorted(set(reconstruction_semantics)),
                },
            )
        )
    else:
        result.append(
            ChangePropagationCapability(
                "datasets.reconstruction",
                ChangePropagationCapabilityStatus.UNAVAILABLE,
                module_paths=(_module_path(provider),)
                if _module_path(provider)
                else (),
                producer_id="kernel-reconstruction@1",
                operations=("reconstruction",),
                diagnostic=_diagnostic(
                    ChangePropagationDiagnosticCode.PARTIAL_INTERFACE
                    if provider_cls is not None
                    else ChangePropagationDiagnosticCode.REQUIRED_SYMBOL_MISSING,
                    "datasets.reconstruction",
                    "no reconstruction-compatible datasets logic backend is fully bound",
                    module=provider_module,
                ),
                reconstruction_compatible=False,
            )
        )
    return tuple(sorted(result, key=lambda item: item.capability_id))


def _probe_datasets_graph_backends(
    *, importer: Callable[[str], Any], timeout_seconds: float
) -> tuple[ChangePropagationCapability, ...]:
    """Adapt exact GraphRAG / Cypher-AST probes; never grant graph authority."""

    provider_module = (
        "ipfs_accelerate_py.agent_supervisor.integrations.ipfs_datasets_analysis_provider"
    )
    capability_ids = ("datasets.graphrag", "datasets.cypher_ast", "datasets.premise_selection")
    completed, provider, error = _bounded_call(
        lambda: importer(provider_module), timeout_seconds
    )
    if not completed or error is not None:
        code = (
            ChangePropagationDiagnosticCode.PROBE_TIMED_OUT
            if not completed
            else ChangePropagationDiagnosticCode.MODULE_IMPORT_FAILED
        )
        status = (
            ChangePropagationCapabilityStatus.TIMED_OUT
            if not completed
            else ChangePropagationCapabilityStatus.UNAVAILABLE
        )
        return tuple(
            ChangePropagationCapability(
                item,
                status,
                producer_id="ipfs-datasets-analysis-provider@1",
                operations=(item.split(".", 1)[-1],),
                diagnostic=_diagnostic(
                    code,
                    item,
                    "datasets graph probe adapter unavailable",
                    module=provider_module,
                    exception=error,
                ),
            )
            for item in capability_ids
        )

    results: list[ChangePropagationCapability] = []
    probe_all = getattr(provider, "probe_all_datasets_graph_backends", None)
    if not callable(probe_all):
        for item in ("datasets.graphrag", "datasets.cypher_ast"):
            results.append(
                ChangePropagationCapability(
                    item,
                    ChangePropagationCapabilityStatus.INCOMPATIBLE,
                    producer_id="ipfs-datasets-analysis-provider@1",
                    operations=(item.split(".", 1)[-1],),
                    diagnostic=_diagnostic(
                        ChangePropagationDiagnosticCode.REQUIRED_SYMBOL_MISSING,
                        item,
                        "datasets graph adapter lacks exact backend probe",
                        module=provider_module,
                    ),
                )
            )
    else:
        completed, probes, error = _bounded_call(
            lambda: probe_all(importer=importer), timeout_seconds
        )
        if not completed or error is not None:
            code = (
                ChangePropagationDiagnosticCode.PROBE_TIMED_OUT
                if not completed
                else ChangePropagationDiagnosticCode.INTERNAL_ERROR
            )
            status = (
                ChangePropagationCapabilityStatus.TIMED_OUT
                if not completed
                else ChangePropagationCapabilityStatus.UNAVAILABLE
            )
            for item in ("datasets.graphrag", "datasets.cypher_ast"):
                results.append(
                    ChangePropagationCapability(
                        item,
                        status,
                        producer_id="ipfs-datasets-analysis-provider@1",
                        operations=(item.split(".", 1)[-1],),
                        diagnostic=_diagnostic(
                            code,
                            item,
                            "datasets graph backend probe failed",
                            module=provider_module,
                            exception=error,
                        ),
                    )
                )
        else:
            for probe in probes:
                kind = str(getattr(getattr(probe, "kind", None), "value", ""))
                capability_id = f"datasets.{kind}"
                receipts = tuple(getattr(probe, "symbol_receipts", ()))
                module_names = tuple(
                    str(getattr(item, "module", ""))
                    for item in receipts
                    if getattr(item, "available", False)
                )
                paths_by_module = _resolve_symbol_paths(
                    module_names, importer=importer, timeout_seconds=timeout_seconds
                )
                paths = tuple(sorted(set(paths_by_module.values())))
                available = (
                    bool(getattr(probe, "available", False))
                    and bool(receipts)
                    and all(bool(getattr(item, "available", False)) for item in receipts)
                    and len(paths_by_module) == len(set(module_names))
                )
                interface = str(getattr(probe, "interface", ""))
                if available:
                    results.append(
                        ChangePropagationCapability(
                            capability_id,
                            ChangePropagationCapabilityStatus.AVAILABLE,
                            module_paths=paths,
                            interface_version=interface,
                            schema_version=str(
                                getattr(provider, "DATASETS_GRAPH_CAPABILITY_SCHEMA", "")
                            ),
                            producer_id=str(
                                getattr(probe, "provider_id", "") or kind
                            ),
                            operations=(kind, "graph_retrieval"),
                            supported_semantics=(
                                kind,
                                "graph_non_authoritative",
                                "nomination_only",
                            ),
                            details={
                                "capability_revision": str(
                                    getattr(probe, "capability_revision", "")
                                ),
                                "package_version": str(
                                    getattr(probe, "package_version", "")
                                ),
                                "authoritative": False,
                            },
                        )
                    )
                else:
                    results.append(
                        ChangePropagationCapability(
                            capability_id,
                            (
                                ChangePropagationCapabilityStatus.PARTIAL
                                if paths
                                else ChangePropagationCapabilityStatus.UNAVAILABLE
                            ),
                            module_paths=paths,
                            interface_version=interface,
                            producer_id=str(
                                getattr(probe, "provider_id", "") or kind
                            ),
                            operations=(kind,),
                            diagnostic=_diagnostic(
                                (
                                    ChangePropagationDiagnosticCode.PARTIAL_INTERFACE
                                    if paths
                                    else ChangePropagationDiagnosticCode.MODULE_IMPORT_FAILED
                                ),
                                capability_id,
                                str(
                                    getattr(
                                        probe,
                                        "unavailable_reason",
                                        "required symbols unavailable",
                                    )
                                ),
                                module=", ".join(module_names),
                            ),
                            details={
                                "reason_code": str(getattr(probe, "reason_code", "")),
                                "authoritative": False,
                            },
                        )
                    )

    # Premise selection is an analysis operation identity, not package presence.
    operation_type = getattr(provider, "AnalysisProviderOperation", None)
    premise_value = ""
    if operation_type is not None:
        premise = getattr(operation_type, "PREMISE_SELECTION", None)
        premise_value = str(getattr(premise, "value", premise) or "")
    provider_path = _module_path(provider)
    if premise_value == "premise_selection" and provider_path:
        results.append(
            ChangePropagationCapability(
                "datasets.premise_selection",
                ChangePropagationCapabilityStatus.AVAILABLE,
                module_paths=(provider_path,),
                interface_version=str(
                    getattr(provider, "BOUNDED_GRAPHRAG_RETRIEVER_INTERFACE_REF", "")
                ),
                schema_version=str(
                    getattr(provider, "DATASETS_GRAPH_CAPABILITY_SCHEMA", "")
                ),
                producer_id="ipfs-datasets-analysis-provider@1",
                operations=("premise_selection", "proof_candidate_selection"),
                supported_semantics=(
                    "premise_selection",
                    "non_authoritative",
                    "bounded_selection",
                ),
                details={"operation": premise_value},
            )
        )
    else:
        results.append(
            ChangePropagationCapability(
                "datasets.premise_selection",
                ChangePropagationCapabilityStatus.UNAVAILABLE,
                module_paths=(provider_path,) if provider_path else (),
                producer_id="ipfs-datasets-analysis-provider@1",
                operations=("premise_selection",),
                diagnostic=_diagnostic(
                    ChangePropagationDiagnosticCode.REQUIRED_SYMBOL_MISSING,
                    "datasets.premise_selection",
                    "AnalysisProviderOperation.PREMISE_SELECTION is not bound",
                    module=provider_module,
                ),
            )
        )
    return tuple(sorted(results, key=lambda item: item.capability_id))


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
) -> ChangePropagationCapability:
    executable = which(command[0])
    if not executable:
        return ChangePropagationCapability(
            capability_id,
            ChangePropagationCapabilityStatus.UNAVAILABLE,
            producer_id=f"toolchain.{command[0]}",
            operations=operations or (command[0],),
            details={"executable_path": "", "version": ""},
            diagnostic=_diagnostic(
                ChangePropagationDiagnosticCode.EXECUTABLE_NOT_FOUND,
                capability_id,
                f"{command[0]} is not on PATH",
            ),
        )
    try:
        # Execute the exact path that was admitted by the locator. This matters
        # for managed/user-site tools whose scripts directory is not on PATH.
        completed = runner(
            (executable, *command[1:]),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return ChangePropagationCapability(
            capability_id,
            ChangePropagationCapabilityStatus.TIMED_OUT,
            producer_id=f"toolchain.{command[0]}",
            operations=operations or (command[0],),
            details={"executable_path": executable, "version": ""},
            diagnostic=_diagnostic(
                ChangePropagationDiagnosticCode.PROBE_TIMED_OUT,
                capability_id,
                "version command exceeded probe timeout",
                exception=exc,
            ),
        )
    except OSError as exc:
        return ChangePropagationCapability(
            capability_id,
            ChangePropagationCapabilityStatus.UNAVAILABLE,
            producer_id=f"toolchain.{command[0]}",
            operations=operations or (command[0],),
            details={"executable_path": executable, "version": ""},
            diagnostic=_diagnostic(
                ChangePropagationDiagnosticCode.EXECUTABLE_VERSION_FAILED,
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
        return ChangePropagationCapability(
            capability_id,
            ChangePropagationCapabilityStatus.UNAVAILABLE,
            producer_id=f"toolchain.{command[0]}",
            operations=operations or (command[0],),
            details={"executable_path": executable, "version": ""},
            diagnostic=_diagnostic(
                ChangePropagationDiagnosticCode.EXECUTABLE_VERSION_FAILED,
                capability_id,
                "version command did not produce a successful version",
                module=executable,
            ),
        )
    version = _first_version(output)
    if expected_version and version != expected_version:
        return ChangePropagationCapability(
            capability_id,
            ChangePropagationCapabilityStatus.INCOMPATIBLE,
            producer_id=f"toolchain.{command[0]}",
            operations=operations or (command[0],),
            details={
                "executable_path": executable,
                "version": version,
                "expected_version": expected_version,
            },
            diagnostic=_diagnostic(
                ChangePropagationDiagnosticCode.EXECUTABLE_VERSION_INCOMPATIBLE,
                capability_id,
                f"expected {expected_version}, got {version or 'unparseable'}",
                module=executable,
            ),
        )
    return ChangePropagationCapability(
        capability_id,
        ChangePropagationCapabilityStatus.AVAILABLE,
        interface_version=version,
        producer_id=f"toolchain.{command[0]}@{version or 'unknown'}",
        operations=operations or (command[0], "version_check"),
        supported_semantics=semantics,
        details={"executable_path": executable, "version_output": output},
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
) -> tuple[str, ChangePropagationCapabilityDiagnostic | None]:
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
            ChangePropagationDiagnosticCode.PROBE_TIMED_OUT
            if isinstance(exc, subprocess.TimeoutExpired)
            else ChangePropagationDiagnosticCode.GITLINK_UNAVAILABLE
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
            ChangePropagationDiagnosticCode.GITLINK_MALFORMED,
            "datasets.gitlink",
            "datasets gitlink is missing or malformed",
        )
    return fields[2], None


def probe_change_propagation_capabilities(
    *,
    importer: Callable[[str], Any] | None = None,
    which: Callable[[str], str | None] | None = None,
    runner: Callable[..., Any] | None = None,
    timeout_seconds: float = DEFAULT_CAPABILITY_PROBE_TIMEOUT_SECONDS,
    repository_root: Path | str | None = None,
) -> ChangePropagationCapabilityReport:
    """Probe exact local interfaces and toolchains without installing or networking.

    Injection points make failures, partial interfaces, and timeouts testable
    without mutating process-wide import state.  A timeout is a diagnosis, not
    an optimistic availability result.  Package presence and solver/model
    candidates grant no authority.
    """

    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds must be a positive number")
    load = importer or importlib.import_module
    # The managed locator is detect-only and additionally checks Python's
    # scripts directory plus the pinned user-local TypeScript toolchain.
    locate = which or find_contract_repair_executable
    execute = runner or subprocess.run
    started = time.monotonic()
    root = (
        Path(repository_root)
        if repository_root is not None
        else Path(__file__).resolve().parents[3]
    )

    capabilities: list[ChangePropagationCapability] = []
    capabilities.extend(
        _probe_interface(spec, importer=load, timeout_seconds=float(timeout_seconds))
        for spec in _INTERFACE_SPECS
    )
    capabilities.extend(
        _probe_datasets_logic_backends(
            importer=load, timeout_seconds=float(timeout_seconds)
        )
    )
    capabilities.extend(
        _probe_datasets_graph_backends(
            importer=load, timeout_seconds=float(timeout_seconds)
        )
    )
    capabilities.extend(
        (
            ChangePropagationCapability(
                "toolchain.python",
                ChangePropagationCapabilityStatus.AVAILABLE,
                interface_version=".".join(map(str, sys.version_info[:3])),
                producer_id=f"python@{'.'.join(map(str, sys.version_info[:3]))}",
                operations=("python_runtime", "type_check_host"),
                supported_semantics=("python_runtime", "host_interpreter"),
                details={
                    "executable_path": sys.executable,
                    "implementation": sys.implementation.name,
                },
            ),
            _run_version(
                "toolchain.node",
                ("node", "--version"),
                which=locate,
                runner=execute,
                timeout_seconds=float(timeout_seconds),
                operations=("node_runtime", "javascript_host"),
                semantics=("node_runtime", "version_checked"),
            ),
            _run_version(
                "toolchain.typescript",
                ("tsc", "--version"),
                expected_version=PINNED_TYPESCRIPT_VERSION,
                which=locate,
                runner=execute,
                timeout_seconds=float(timeout_seconds),
                operations=("typescript_compile", "tsc"),
                semantics=("typescript", "pinned_version", "version_checked"),
            ),
            _run_version(
                "toolchain.mypy",
                ("mypy", "--version"),
                which=locate,
                runner=execute,
                timeout_seconds=float(timeout_seconds),
                operations=("mypy_typecheck",),
                semantics=("mypy", "static_types", "version_checked"),
            ),
            _run_version(
                "toolchain.cvc5",
                ("cvc5", "--version"),
                expected_version=PINNED_CVC5_VERSION,
                which=locate,
                runner=execute,
                timeout_seconds=float(timeout_seconds),
                operations=("smt_solve", "cvc5"),
                semantics=(
                    "cvc5",
                    "solver_candidates_non_authoritative",
                    "version_checked",
                ),
            ),
            _run_version(
                "toolchain.z3",
                ("z3", "--version"),
                which=locate,
                runner=execute,
                timeout_seconds=float(timeout_seconds),
                operations=("smt_solve", "z3"),
                semantics=(
                    "z3",
                    "solver_candidates_non_authoritative",
                    "version_checked",
                ),
            ),
        )
    )
    gitlink_revision, gitlink_diagnostic = _gitlink_revision(
        root, execute, float(timeout_seconds)
    )
    if gitlink_diagnostic:
        capabilities.append(
            ChangePropagationCapability(
                "datasets.gitlink",
                ChangePropagationCapabilityStatus.UNAVAILABLE,
                producer_id="datasets-gitlink@1",
                operations=("gitlink_revision",),
                diagnostic=gitlink_diagnostic,
            )
        )
    else:
        capabilities.append(
            ChangePropagationCapability(
                "datasets.gitlink",
                ChangePropagationCapabilityStatus.AVAILABLE,
                producer_id="datasets-gitlink@1",
                operations=("gitlink_revision",),
                supported_semantics=("gitlink_revision_bound",),
                details={
                    "executable_path": "git",
                    "revision": gitlink_revision,
                },
            )
        )

    all_diagnostics = tuple(
        item.diagnostic for item in capabilities if item.diagnostic is not None
    )
    datasets_paths = tuple(
        sorted(
            {
                path
                for item in capabilities
                if item.capability_id.startswith("datasets.")
                for path in item.module_paths
            }
        )
    )
    accelerator_prefixes = (
        "index.",
        "graph.",
        "vector.",
        "llm.",
        "datasets.analysis_provider",
        "datasets.logic_provider",
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
    return ChangePropagationCapabilityReport(
        capabilities=tuple(sorted(capabilities, key=lambda item: item.capability_id)),
        accelerator_module_paths=accelerator_paths,
        datasets_module_paths=datasets_paths,
        datasets_gitlink_revision=gitlink_revision,
        diagnostics=all_diagnostics,
        generated_at_monotonic=started,
        duration_seconds=time.monotonic() - started,
    )


__all__ = [
    "CHANGE_PROPAGATION_CAPABILITY_REPORT_SCHEMA_VERSION",
    "CHANGE_PROPAGATION_CAPABILITY_REPORT_VERSION",
    "CHANGE_PROPAGATION_CAPABILITY_SCHEMA_VERSION",
    "DEFAULT_CAPABILITY_PROBE_TIMEOUT_SECONDS",
    "PINNED_CVC5_VERSION",
    "PINNED_TYPESCRIPT_VERSION",
    "ChangePropagationCapability",
    "ChangePropagationCapabilityDiagnostic",
    "ChangePropagationCapabilityReport",
    "ChangePropagationCapabilityStatus",
    "ChangePropagationDiagnosticCode",
    "probe_change_propagation_capabilities",
]
