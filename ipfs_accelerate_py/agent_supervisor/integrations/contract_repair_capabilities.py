"""Lazy, fail-closed capability admission for proof-gated contract repair.

Importing this module is deliberately cheap: optional datasets logic modules,
VFS work-in-progress modules, and executables are inspected only by
``probe_contract_repair_capabilities``.  Discovery is not authority.  In
particular, a package, a module spec, or a solver binary is never admitted
until its exact interface is imported/checked (or its version command has
completed) within the probe budget.

The probe never installs packages, invokes package managers, or contacts a
network service.  Solver output is not requested; an executable capability is
only a route that may produce a non-authoritative candidate which still needs
independent reconstruction.
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final


CONTRACT_REPAIR_CAPABILITY_SCHEMA_VERSION: Final = (
    "ipfs_accelerate_py/agent-supervisor/contract-repair-capability@1"
)
CONTRACT_REPAIR_CAPABILITY_REPORT_SCHEMA_VERSION: Final = (
    "ipfs_accelerate_py/agent-supervisor/contract-repair-capability-report@1"
)
CONTRACT_REPAIR_CAPABILITY_REPORT_VERSION: Final = 1
DEFAULT_CAPABILITY_PROBE_TIMEOUT_SECONDS: Final = 2.0
PINNED_TYPESCRIPT_VERSION: Final = "5.6.3"


class ContractRepairCapabilityStatus(str, Enum):
    """Closed admission outcomes; only ``AVAILABLE`` admits an interface."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    UNSUPPORTED = "unsupported"
    INCOMPATIBLE = "incompatible"
    PARTIAL = "partial"
    TIMED_OUT = "timed_out"


class ContractRepairDiagnosticCode(str, Enum):
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
class ContractRepairCapabilityDiagnostic:
    """Typed, bounded probe diagnostic rather than an exception-only failure."""

    code: ContractRepairDiagnosticCode
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
class ContractRepairCapability:
    """One exact capability binding used by the repair admission gate."""

    capability_id: str
    status: ContractRepairCapabilityStatus
    module_paths: tuple[str, ...] = ()
    interface_version: str = ""
    schema_version: str = ""
    supported_semantics: tuple[str, ...] = ()
    diagnostic: ContractRepairCapabilityDiagnostic | None = None
    reconstruction_compatible: bool = False
    candidate_authoritative: bool = False
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.capability_id.strip():
            raise ValueError("capability_id must not be empty")
        if self.status is ContractRepairCapabilityStatus.AVAILABLE:
            if not self.module_paths and not self.details.get("executable_path"):
                raise ValueError("available capability requires an exact module or executable path")
            if self.diagnostic is not None:
                raise ValueError("available capability cannot carry a failure diagnostic")
        elif self.diagnostic is None:
            raise ValueError("non-available capability requires a typed diagnostic")
        if self.candidate_authoritative:
            raise ValueError("solver and analysis candidates cannot be authoritative")
        object.__setattr__(self, "module_paths", tuple(sorted(set(self.module_paths))))
        object.__setattr__(self, "supported_semantics", tuple(sorted(set(self.supported_semantics))))
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))

    @property
    def available(self) -> bool:
        return self.status is ContractRepairCapabilityStatus.AVAILABLE

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
            "supported_semantics": list(self.supported_semantics),
            "reconstruction_compatible": self.reconstruction_compatible,
            "candidate_authoritative": False,
            "diagnostic": self.diagnostic.to_dict() if self.diagnostic else None,
            "details": dict(self.details),
        }


# Compatibility spelling used by the datasets/VFS plans.  It intentionally is
# a type alias, not a second less-strict record shape.
BackendCapability = ContractRepairCapability


@dataclass(frozen=True)
class ContractRepairCapabilityReport:
    """Immutable, versioned snapshot of all contract-repair prerequisites."""

    capabilities: tuple[ContractRepairCapability, ...]
    accelerator_module_paths: tuple[str, ...]
    datasets_module_paths: tuple[str, ...]
    datasets_gitlink_revision: str
    diagnostics: tuple[ContractRepairCapabilityDiagnostic, ...] = ()
    generated_at_monotonic: float = 0.0
    duration_seconds: float = 0.0
    schema_version: str = CONTRACT_REPAIR_CAPABILITY_REPORT_SCHEMA_VERSION
    report_version: int = CONTRACT_REPAIR_CAPABILITY_REPORT_VERSION

    def __post_init__(self) -> None:
        ids = [item.capability_id for item in self.capabilities]
        if len(ids) != len(set(ids)):
            raise ValueError("capability ids must be unique")
        if self.report_version != CONTRACT_REPAIR_CAPABILITY_REPORT_VERSION:
            raise ValueError("unsupported capability report version")
        if self.duration_seconds < 0:
            raise ValueError("duration_seconds must be non-negative")
        object.__setattr__(self, "capabilities", tuple(self.capabilities))
        object.__setattr__(self, "accelerator_module_paths", tuple(sorted(set(self.accelerator_module_paths))))
        object.__setattr__(self, "datasets_module_paths", tuple(sorted(set(self.datasets_module_paths))))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))

    @property
    def capability_map(self) -> Mapping[str, ContractRepairCapability]:
        return MappingProxyType({item.capability_id: item for item in self.capabilities})

    # A concise compatibility spelling for callers following other capability
    # reports in this package.
    @property
    def capabilities_by_id(self) -> Mapping[str, ContractRepairCapability]:
        return self.capability_map

    def capability(self, capability_id: str) -> ContractRepairCapability:
        try:
            return self.capability_map[capability_id]
        except KeyError as exc:
            raise KeyError(f"unknown contract-repair capability: {capability_id}") from exc

    @property
    def toolchains(self) -> Mapping[str, ContractRepairCapability]:
        return MappingProxyType(
            {
                item.capability_id: item
                for item in self.capabilities
                if item.capability_id.startswith("toolchain.")
            }
        )

    @property
    def gitlink_revision(self) -> str:
        """Compatibility spelling for the datasets submodule gitlink."""

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
    semantics: tuple[str, ...] = ()


_VFS_INTERFACE_SPECS: Final = (
    _InterfaceSpec(
        "accelerator.ipfs_datasets_logic_provider",
        "ipfs_accelerate_py.agent_supervisor.integrations.ipfs_datasets_logic_provider",
        ("IpfsDatasetsLogicProvider", "DatasetsLogicBackendProbe", "probe_all_datasets_logic_backends"),
        interface_constant="IPFS_DATASETS_LOGIC_PROVIDER_VERSION",
        schema_constant="HAMMER_ADAPTER_SCHEMA_VERSION",
        semantics=("IPFSDatasetsLogicProvider", "BackendCapability", "lazy_hammer_adapter"),
    ),
    _InterfaceSpec("vfs.program_graph", "ipfs_accelerate_py.agent_supervisor.program_graph", ("ProgramGraph",), semantics=("program_graph", "complete_frontier")),
    _InterfaceSpec("vfs.program_call_resolver", "ipfs_accelerate_py.agent_supervisor.program_call_resolver", ("ProgramCallResolver",), semantics=("conservative_call_resolution", "unknown_frontier")),
    _InterfaceSpec("vfs.contract_extractor", "ipfs_accelerate_py.agent_supervisor.contract_extractor", ("ContractExtractor",), semantics=("contract_extraction",)),
    _InterfaceSpec("vfs.contract_checker", "ipfs_accelerate_py.agent_supervisor.contract_checker", ("ContractChecker",), semantics=("symbolic_contract_comparison", "counterexamples")),
    _InterfaceSpec("vfs.contract_prover", "ipfs_accelerate_py.agent_supervisor.code_contract_prover", ("CodeContractProver",), semantics=("proof_obligation_routing",)),
    _InterfaceSpec("vfs.repair", "ipfs_accelerate_py.agent_supervisor.contract_repair_packet", ("ContractRepairPacket",), semantics=("bounded_repair_packet",)),
    _InterfaceSpec(
        "vfs.program_contract",
        "ipfs_accelerate_py.agent_supervisor.program_contracts",
        ("ExpectedProgramContract", "ObservedProgramContract", "ProgramContractBundle"),
        interface_constant="PROGRAM_CONTRACT_VERSION",
        expected_interface="1",
        schema_constant="SCHEMA_VERSION",
        expected_schema="1",
        semantics=("ProgramContract@1",),
    ),
)


def _diagnostic(
    code: ContractRepairDiagnosticCode,
    capability_id: str,
    message: str,
    *,
    module: str = "",
    exception: BaseException | None = None,
) -> ContractRepairCapabilityDiagnostic:
    return ContractRepairCapabilityDiagnostic(
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
) -> ContractRepairCapability:
    completed, module, error = _bounded_call(lambda: importer(spec.module), timeout_seconds)
    if not completed:
        return ContractRepairCapability(
            spec.capability_id, ContractRepairCapabilityStatus.TIMED_OUT,
            diagnostic=_diagnostic(ContractRepairDiagnosticCode.PROBE_TIMED_OUT, spec.capability_id, "module import exceeded probe timeout", module=spec.module),
        )
    if error is not None:
        return ContractRepairCapability(
            spec.capability_id, ContractRepairCapabilityStatus.UNAVAILABLE,
            diagnostic=_diagnostic(ContractRepairDiagnosticCode.MODULE_IMPORT_FAILED, spec.capability_id, "required module could not be imported", module=spec.module, exception=error),
        )
    path = _module_path(module)
    if not path:
        return ContractRepairCapability(
            spec.capability_id, ContractRepairCapabilityStatus.INCOMPATIBLE,
            diagnostic=_diagnostic(ContractRepairDiagnosticCode.MODULE_PATH_UNAVAILABLE, spec.capability_id, "imported module has no exact file path", module=spec.module),
        )
    missing = [symbol for symbol in spec.symbols if not hasattr(module, symbol)]
    if missing:
        return ContractRepairCapability(
            spec.capability_id, ContractRepairCapabilityStatus.PARTIAL, module_paths=(path,),
            diagnostic=_diagnostic(ContractRepairDiagnosticCode.REQUIRED_SYMBOL_MISSING, spec.capability_id, f"missing required interface symbols: {', '.join(missing)}", module=spec.module),
        )
    interface_version = str(getattr(module, spec.interface_constant, "")) if spec.interface_constant else ""
    if spec.expected_interface and interface_version != spec.expected_interface:
        return ContractRepairCapability(
            spec.capability_id, ContractRepairCapabilityStatus.INCOMPATIBLE, module_paths=(path,), interface_version=interface_version,
            diagnostic=_diagnostic(ContractRepairDiagnosticCode.INTERFACE_VERSION_INCOMPATIBLE, spec.capability_id, f"expected interface version {spec.expected_interface!r}, got {interface_version!r}", module=spec.module),
        )
    schema_version = str(getattr(module, spec.schema_constant, "")) if spec.schema_constant else ""
    if spec.expected_schema and schema_version != spec.expected_schema:
        return ContractRepairCapability(
            spec.capability_id, ContractRepairCapabilityStatus.INCOMPATIBLE, module_paths=(path,), interface_version=interface_version, schema_version=schema_version,
            diagnostic=_diagnostic(ContractRepairDiagnosticCode.SCHEMA_VERSION_INCOMPATIBLE, spec.capability_id, f"expected schema version {spec.expected_schema!r}, got {schema_version!r}", module=spec.module),
        )
    return ContractRepairCapability(
        spec.capability_id, ContractRepairCapabilityStatus.AVAILABLE, module_paths=(path,), interface_version=interface_version, schema_version=schema_version, supported_semantics=spec.semantics,
    )


def _probe_datasets_backends(
    *, importer: Callable[[str], Any], timeout_seconds: float
) -> tuple[ContractRepairCapability, ...]:
    """Adapt the existing exact-symbol datasets probe without trusting labels."""

    provider_module = "ipfs_accelerate_py.agent_supervisor.integrations.ipfs_datasets_logic_provider"
    completed, provider, error = _bounded_call(lambda: importer(provider_module), timeout_seconds)
    capability_ids = ("datasets.logic_ir", "datasets.tdfol", "datasets.cec", "datasets.smt", "datasets.hammer")
    if not completed or error is not None:
        code = ContractRepairDiagnosticCode.PROBE_TIMED_OUT if not completed else ContractRepairDiagnosticCode.MODULE_IMPORT_FAILED
        status = ContractRepairCapabilityStatus.TIMED_OUT if not completed else ContractRepairCapabilityStatus.UNAVAILABLE
        return tuple(
            ContractRepairCapability(item, status, diagnostic=_diagnostic(code, item, "datasets logic probe adapter unavailable", module=provider_module, exception=error))
            for item in capability_ids
        )
    probe_all = getattr(provider, "probe_all_datasets_logic_backends", None)
    kind_type = getattr(provider, "DatasetsLogicBackendKind", None)
    if not callable(probe_all) or kind_type is None:
        return tuple(
            ContractRepairCapability(item, ContractRepairCapabilityStatus.INCOMPATIBLE, diagnostic=_diagnostic(ContractRepairDiagnosticCode.REQUIRED_SYMBOL_MISSING, item, "datasets logic adapter lacks exact backend probe", module=provider_module))
            for item in capability_ids
        )
    completed, probes, error = _bounded_call(lambda: probe_all(importer=importer), timeout_seconds)
    if not completed or error is not None:
        code = ContractRepairDiagnosticCode.PROBE_TIMED_OUT if not completed else ContractRepairDiagnosticCode.INTERNAL_ERROR
        status = ContractRepairCapabilityStatus.TIMED_OUT if not completed else ContractRepairCapabilityStatus.UNAVAILABLE
        return tuple(
            ContractRepairCapability(item, status, diagnostic=_diagnostic(code, item, "datasets logic backend probe failed", module=provider_module, exception=error))
            for item in capability_ids
        )
    result: list[ContractRepairCapability] = []
    for probe in probes:
        kind = str(getattr(getattr(probe, "kind", None), "value", ""))
        capability_id = f"datasets.{ 'logic_ir' if kind == 'ir' else kind }"
        receipts = tuple(getattr(probe, "symbol_receipts", ()))
        module_names = tuple(str(getattr(item, "module", "")) for item in receipts if getattr(item, "available", False))
        # The normal import path leaves the checked modules in ``sys.modules``.
        # Injected importers used by embedded deployments/tests need not do so,
        # however, so re-read only the already admitted exact module names to
        # bind their physical paths.  This is still bounded and does not turn a
        # package marker into an available backend.
        paths_by_module: dict[str, str] = {}
        for module_name in module_names:
            checked_module = sys.modules.get(module_name)
            if checked_module is None:
                completed_path, checked_module, path_error = _bounded_call(
                    lambda module_name=module_name: importer(module_name), timeout_seconds
                )
                if not completed_path or path_error is not None:
                    continue
            path = _module_path(checked_module)
            if path:
                paths_by_module[module_name] = path
        paths = tuple(sorted(set(paths_by_module.values())))
        available = (
            bool(getattr(probe, "available", False))
            and bool(receipts)
            and all(bool(getattr(item, "available", False)) for item in receipts)
            and len(paths_by_module) == len(set(module_names))
        )
        if available:
            result.append(ContractRepairCapability(
                capability_id, ContractRepairCapabilityStatus.AVAILABLE, module_paths=paths,
                interface_version=str(getattr(provider, "LOGIC_IR_INTERFACE", "")),
                schema_version=str(getattr(provider, "DATASETS_LOGIC_PROBE_SCHEMA", "")),
                supported_semantics=(kind, "solver_candidates_non_authoritative", "independent_reconstruction_required"),
                reconstruction_compatible=bool(getattr(probe, "reconstruction_compatible", False)),
                details={"capability_revision": str(getattr(probe, "capability_revision", "")), "package_version": str(getattr(probe, "package_version", "")), "provider_id": str(getattr(probe, "provider_id", ""))},
            ))
        else:
            result.append(ContractRepairCapability(
                capability_id, ContractRepairCapabilityStatus.PARTIAL if paths else ContractRepairCapabilityStatus.UNAVAILABLE, module_paths=paths,
                diagnostic=_diagnostic(ContractRepairDiagnosticCode.PARTIAL_INTERFACE if paths else ContractRepairDiagnosticCode.MODULE_IMPORT_FAILED, capability_id, str(getattr(probe, "unavailable_reason", "required symbols unavailable")), module=", ".join(module_names)),
                reconstruction_compatible=bool(getattr(probe, "reconstruction_compatible", False)),
                details={"reason_code": str(getattr(probe, "reason_code", ""))},
            ))
    return tuple(sorted(result, key=lambda item: item.capability_id))


def _run_version(
    capability_id: str,
    command: tuple[str, ...],
    *,
    expected_version: str = "",
    which: Callable[[str], str | None],
    runner: Callable[..., Any],
    timeout_seconds: float,
) -> ContractRepairCapability:
    executable = which(command[0])
    if not executable:
        return ContractRepairCapability(capability_id, ContractRepairCapabilityStatus.UNAVAILABLE, details={"executable_path": "", "version": ""}, diagnostic=_diagnostic(ContractRepairDiagnosticCode.EXECUTABLE_NOT_FOUND, capability_id, f"{command[0]} is not on PATH"))
    try:
        completed = runner(command, capture_output=True, text=True, timeout=timeout_seconds, check=False)
    except subprocess.TimeoutExpired as exc:
        return ContractRepairCapability(capability_id, ContractRepairCapabilityStatus.TIMED_OUT, details={"executable_path": executable, "version": ""}, diagnostic=_diagnostic(ContractRepairDiagnosticCode.PROBE_TIMED_OUT, capability_id, "version command exceeded probe timeout", exception=exc))
    except OSError as exc:
        return ContractRepairCapability(capability_id, ContractRepairCapabilityStatus.UNAVAILABLE, details={"executable_path": executable, "version": ""}, diagnostic=_diagnostic(ContractRepairDiagnosticCode.EXECUTABLE_VERSION_FAILED, capability_id, "version command could not run", exception=exc))
    output = ((getattr(completed, "stdout", "") or "") + "\n" + (getattr(completed, "stderr", "") or "")).strip()
    if getattr(completed, "returncode", 1) != 0 or not output:
        return ContractRepairCapability(capability_id, ContractRepairCapabilityStatus.UNAVAILABLE, details={"executable_path": executable, "version": ""}, diagnostic=_diagnostic(ContractRepairDiagnosticCode.EXECUTABLE_VERSION_FAILED, capability_id, "version command did not produce a successful version", module=executable))
    version = _first_version(output)
    if expected_version and version != expected_version:
        return ContractRepairCapability(capability_id, ContractRepairCapabilityStatus.INCOMPATIBLE, details={"executable_path": executable, "version": version, "expected_version": expected_version}, diagnostic=_diagnostic(ContractRepairDiagnosticCode.EXECUTABLE_VERSION_INCOMPATIBLE, capability_id, f"expected {expected_version}, got {version or 'unparseable'}", module=executable))
    return ContractRepairCapability(capability_id, ContractRepairCapabilityStatus.AVAILABLE, interface_version=version, supported_semantics=("version_checked",), details={"executable_path": executable, "version_output": output})


def _first_version(value: str) -> str:
    for token in value.replace("\n", " ").split():
        normalized = token.strip("vV,;()[]")
        if normalized and normalized[0].isdigit() and all(part.isdigit() for part in normalized.split(".") if part):
            return normalized
    return ""


def _gitlink_revision(root: Path, runner: Callable[..., Any], timeout_seconds: float) -> tuple[str, ContractRepairCapabilityDiagnostic | None]:
    try:
        completed = runner(("git", "-C", str(root), "ls-tree", "HEAD", "--", "ipfs_datasets_py"), capture_output=True, text=True, timeout=timeout_seconds, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        code = ContractRepairDiagnosticCode.PROBE_TIMED_OUT if isinstance(exc, subprocess.TimeoutExpired) else ContractRepairDiagnosticCode.GITLINK_UNAVAILABLE
        return "", _diagnostic(code, "datasets.gitlink", "could not read datasets gitlink", exception=exc)
    output = (getattr(completed, "stdout", "") or "").strip()
    fields = output.split()
    if getattr(completed, "returncode", 1) != 0 or len(fields) < 3 or fields[0] != "160000" or len(fields[2]) != 40:
        return "", _diagnostic(ContractRepairDiagnosticCode.GITLINK_MALFORMED, "datasets.gitlink", "datasets gitlink is missing or malformed")
    return fields[2], None


def probe_contract_repair_capabilities(
    *,
    importer: Callable[[str], Any] | None = None,
    which: Callable[[str], str | None] | None = None,
    runner: Callable[..., Any] | None = None,
    timeout_seconds: float = DEFAULT_CAPABILITY_PROBE_TIMEOUT_SECONDS,
    repository_root: Path | str | None = None,
) -> ContractRepairCapabilityReport:
    """Probe exact local interfaces and toolchains without installing or networking.

    Injection points make failures, partial interfaces, and timeouts testable
    without mutating process-wide import state.  A timeout is a diagnosis, not
    an optimistic availability result.
    """

    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be a positive number")
    load = importer or importlib.import_module
    locate = which or shutil.which
    execute = runner or subprocess.run
    started = time.monotonic()
    root = Path(repository_root) if repository_root is not None else Path(__file__).resolve().parents[3]

    capabilities = list(_probe_datasets_backends(importer=load, timeout_seconds=float(timeout_seconds)))
    capabilities.extend(_probe_interface(spec, importer=load, timeout_seconds=float(timeout_seconds)) for spec in _VFS_INTERFACE_SPECS)
    capabilities.extend((
        ContractRepairCapability("toolchain.python", ContractRepairCapabilityStatus.AVAILABLE, interface_version=".".join(map(str, sys.version_info[:3])), supported_semantics=("python_runtime",), details={"executable_path": sys.executable, "implementation": sys.implementation.name}),
        _run_version("toolchain.node", ("node", "--version"), which=locate, runner=execute, timeout_seconds=float(timeout_seconds)),
        _run_version("toolchain.typescript", ("tsc", "--version"), expected_version=PINNED_TYPESCRIPT_VERSION, which=locate, runner=execute, timeout_seconds=float(timeout_seconds)),
        _run_version("toolchain.mypy", ("mypy", "--version"), which=locate, runner=execute, timeout_seconds=float(timeout_seconds)),
        _run_version("toolchain.cvc5", ("cvc5", "--version"), which=locate, runner=execute, timeout_seconds=float(timeout_seconds)),
        _run_version("toolchain.z3", ("z3", "--version"), which=locate, runner=execute, timeout_seconds=float(timeout_seconds)),
    ))
    gitlink_revision, gitlink_diagnostic = _gitlink_revision(root, execute, float(timeout_seconds))
    if gitlink_diagnostic:
        capabilities.append(ContractRepairCapability("datasets.gitlink", ContractRepairCapabilityStatus.UNAVAILABLE, diagnostic=gitlink_diagnostic))
    else:
        capabilities.append(ContractRepairCapability("datasets.gitlink", ContractRepairCapabilityStatus.AVAILABLE, supported_semantics=("gitlink_revision_bound",), details={"executable_path": "git", "revision": gitlink_revision}))
    all_diagnostics = tuple(item.diagnostic for item in capabilities if item.diagnostic is not None)
    datasets_paths = tuple(sorted({path for item in capabilities if item.capability_id.startswith("datasets.") for path in item.module_paths}))
    accelerator_paths = tuple(sorted({path for item in capabilities if item.capability_id.startswith(("vfs.", "accelerator.")) for path in item.module_paths} | {os.path.realpath(__file__)}))
    return ContractRepairCapabilityReport(
        capabilities=tuple(sorted(capabilities, key=lambda item: item.capability_id)),
        accelerator_module_paths=accelerator_paths,
        datasets_module_paths=datasets_paths,
        datasets_gitlink_revision=gitlink_revision,
        diagnostics=all_diagnostics,
        generated_at_monotonic=started,
        duration_seconds=time.monotonic() - started,
    )


__all__ = [
    "BackendCapability",
    "CONTRACT_REPAIR_CAPABILITY_REPORT_SCHEMA_VERSION",
    "CONTRACT_REPAIR_CAPABILITY_REPORT_VERSION",
    "CONTRACT_REPAIR_CAPABILITY_SCHEMA_VERSION",
    "DEFAULT_CAPABILITY_PROBE_TIMEOUT_SECONDS",
    "PINNED_TYPESCRIPT_VERSION",
    "ContractRepairCapability",
    "ContractRepairCapabilityDiagnostic",
    "ContractRepairCapabilityReport",
    "ContractRepairCapabilityStatus",
    "ContractRepairDiagnosticCode",
    "probe_contract_repair_capabilities",
]
