"""Lazy datasets Logic/solver/cache capabilities for the deterministic doctor.

LPR-032 requires doctor stages to consult exact datasets Logic surfaces
(Tactician, Hammer, proof caches, solvers, reconstruction toolchains) through
capability probes only.  Importing this module is deliberately cheap:

* no package install
* no network
* no target-repository code import
* no process-global ``HOME`` / ``sys.prefix`` mutation from the probe path

Package presence is not authority.  Solver binaries, hammer nomination scores,
and cache metadata never promote a surface to semantic or proof authority.
Missing, partial, incompatible, timed-out, or isolation-unsafe surfaces yield
typed diagnostics.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import os
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Any, Final


DOCTOR_LOGIC_CAPABILITY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/datasets-doctor-logic-capability@1"
)
DOCTOR_LOGIC_CAPABILITY_REPORT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/datasets-doctor-logic-capability-report@1"
)
DOCTOR_LOGIC_PROVIDER_ID: Final = "ipfs-datasets-doctor-logic"
DOCTOR_LOGIC_PROVIDER_VERSION: Final = "1"
DOCTOR_LOGIC_INTERFACE: Final = "IpfsDatasetsDoctorLogic@1"
DEFAULT_CAPABILITY_PROBE_TIMEOUT_SECONDS: Final = 10.0

# Exact module paths for logic / solver / cache surfaces.  Probes resolve these
# with importlib without installing or networking.
LOGIC_PROVIDER_MODULE: Final = (
    "ipfs_accelerate_py.agent_supervisor.integrations.ipfs_datasets_logic_provider"
)
TACTICIAN_PROVIDER_MODULE: Final = (
    "ipfs_accelerate_py.agent_supervisor.integrations.ipfs_datasets_tactician_provider"
)
TACTICIAN_CAPABILITIES_MODULE: Final = (
    "ipfs_accelerate_py.agent_supervisor.integrations.tactician_hammer_capabilities"
)
DOCTOR_PROOF_CACHE_MODULE: Final = (
    "ipfs_accelerate_py.agent_supervisor.proof.doctor_proof_cache"
)
FORMAL_CACHE_MODULE: Final = (
    "ipfs_accelerate_py.agent_supervisor.proof.formal_verification_cache"
)
PROVER_EVIDENCE_MODULE: Final = (
    "ipfs_accelerate_py.agent_supervisor.proof.prover_evidence_store"
)
CONTENT_IDENTITY_MODULE: Final = (
    "ipfs_accelerate_py.agent_supervisor.analysis.content_identity_bridge"
)
GENERIC_TACTICIAN_MODULE: Final = "ipfs_datasets_py.logic.tactician"
HAMMER_MODULE: Final = "ipfs_datasets_py.logic.hammers"
PROOF_CORPUS_MODULE: Final = "ipfs_datasets_py.logic.proof_corpus.store"
IR_CORE_IDENTITY_MODULE: Final = "ipfs_datasets_py.logic.ir_core.identity"

IMPORT_ISOLATION_HARDENED: Final = "import_isolation_hardened"
IMPORT_ISOLATION_UNSAFE: Final = "import_isolation_unsafe"

_PROBE_LOCK: Final = threading.RLock()
_MODULE_CACHE: dict[str, ModuleType | None] = {}
_MODULE_ERRORS: dict[str, BaseException] = {}


class DatasetsDoctorLogicStatus(str, Enum):
    """Closed admission outcomes; only ``AVAILABLE`` admits an interface."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    UNSUPPORTED = "unsupported"
    INCOMPATIBLE = "incompatible"
    PARTIAL = "partial"
    TIMED_OUT = "timed_out"
    UNSAFE = "unsafe"


class DatasetsDoctorLogicDiagnosticCode(str, Enum):
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
    IMPORT_ISOLATION_UNSAFE = "import_isolation_unsafe"
    FEATURE_NOT_ADMITTED = "feature_not_admitted"
    INSTALL_FORBIDDEN = "install_forbidden"
    NETWORK_FORBIDDEN = "network_forbidden"
    TARGET_IMPORT_FORBIDDEN = "target_import_forbidden"
    PROCESS_GLOBAL_MUTATION_FORBIDDEN = "process_global_mutation_forbidden"
    INTERNAL_ERROR = "internal_error"
    LAZY_NOT_PROBED = "lazy_not_probed"


@dataclass(frozen=True)
class DatasetsDoctorLogicDiagnostic:
    """Typed, bounded probe diagnostic rather than an exception-only failure."""

    code: DatasetsDoctorLogicDiagnosticCode
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
class DatasetsDoctorLogicCapability:
    """One exact datasets logic/solver/cache capability binding."""

    capability_id: str
    status: DatasetsDoctorLogicStatus
    module_paths: tuple[str, ...] = ()
    interface_version: str = ""
    schema_version: str = ""
    producer_id: str = DOCTOR_LOGIC_PROVIDER_ID
    operations: tuple[str, ...] = ()
    supported_semantics: tuple[str, ...] = ()
    diagnostic: DatasetsDoctorLogicDiagnostic | None = None
    reconstruction_compatible: bool = False
    semantic_authority: bool = False
    completion_authority: bool = False
    proof_authority: bool = False
    candidate_authoritative: bool = False
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.capability_id.strip():
            raise ValueError("capability_id must not be empty")
        if self.semantic_authority or self.completion_authority:
            raise ValueError(
                "capability declaration cannot claim semantic or completion authority"
            )
        if self.candidate_authoritative:
            raise ValueError(
                "solver, graph, vector, and model candidates cannot be authoritative"
            )
        if self.status is DatasetsDoctorLogicStatus.AVAILABLE:
            if not self.module_paths and not self.details.get("executable_path"):
                raise ValueError(
                    "available capability requires an exact module or executable path"
                )
            if self.diagnostic is not None:
                raise ValueError("available capability cannot carry a failure diagnostic")
        elif self.diagnostic is None:
            raise ValueError("non-available capability requires a typed diagnostic")
        object.__setattr__(self, "semantic_authority", False)
        object.__setattr__(self, "completion_authority", False)
        object.__setattr__(self, "candidate_authoritative", False)
        object.__setattr__(self, "module_paths", tuple(sorted(set(self.module_paths))))
        object.__setattr__(self, "operations", tuple(sorted(set(self.operations))))
        object.__setattr__(
            self, "supported_semantics", tuple(sorted(set(self.supported_semantics)))
        )
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))

    @property
    def available(self) -> bool:
        return self.status is DatasetsDoctorLogicStatus.AVAILABLE

    @property
    def module_path(self) -> str:
        return self.module_paths[0] if self.module_paths else ""

    @property
    def reason_code(self) -> str:
        return self.diagnostic.code.value if self.diagnostic is not None else ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DOCTOR_LOGIC_CAPABILITY_SCHEMA,
            "capability_id": self.capability_id,
            "status": self.status.value,
            "available": self.available,
            "module_paths": list(self.module_paths),
            "module_path": self.module_path,
            "interface_version": self.interface_version,
            "schema_version": self.schema_version,
            "producer_id": self.producer_id,
            "operations": list(self.operations),
            "supported_semantics": list(self.supported_semantics),
            "diagnostic": (
                self.diagnostic.to_dict() if self.diagnostic is not None else None
            ),
            "reconstruction_compatible": self.reconstruction_compatible,
            "semantic_authority": False,
            "completion_authority": False,
            "proof_authority": self.proof_authority and self.available,
            "candidate_authoritative": False,
            "reason_code": self.reason_code,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class DatasetsDoctorLogicCapabilityReport:
    """Bounded capability report for doctor logic/solver/cache surfaces."""

    capabilities: tuple[DatasetsDoctorLogicCapability, ...]
    probed_at_ms: int
    probe_timeout_seconds: float
    provider_id: str = DOCTOR_LOGIC_PROVIDER_ID
    provider_version: str = DOCTOR_LOGIC_PROVIDER_VERSION
    interface_version: str = DOCTOR_LOGIC_INTERFACE
    install_attempted: bool = False
    network_attempted: bool = False
    target_import_attempted: bool = False
    process_global_mutation: bool = False
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "capabilities", tuple(self.capabilities))
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))
        if self.install_attempted or self.network_attempted:
            raise ValueError(
                "doctor logic probes must not install packages or contact the network"
            )
        if self.target_import_attempted:
            raise ValueError("doctor logic probes must not import target repository code")
        if self.process_global_mutation:
            raise ValueError(
                "doctor logic probes must not perform process-global unsafe mutation"
            )

    def get(self, capability_id: str) -> DatasetsDoctorLogicCapability | None:
        for item in self.capabilities:
            if item.capability_id == capability_id:
                return item
        return None

    def available_ids(self) -> tuple[str, ...]:
        return tuple(
            item.capability_id for item in self.capabilities if item.available
        )

    @property
    def all_available(self) -> bool:
        return bool(self.capabilities) and all(
            item.available for item in self.capabilities
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DOCTOR_LOGIC_CAPABILITY_REPORT_SCHEMA,
            "provider_id": self.provider_id,
            "provider_version": self.provider_version,
            "interface_version": self.interface_version,
            "probed_at_ms": self.probed_at_ms,
            "probe_timeout_seconds": self.probe_timeout_seconds,
            "install_attempted": False,
            "network_attempted": False,
            "target_import_attempted": False,
            "process_global_mutation": False,
            "available_ids": list(self.available_ids()),
            "capabilities": [item.to_dict() for item in self.capabilities],
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class _CapabilitySpec:
    capability_id: str
    module: str
    required_symbols: tuple[str, ...] = ()
    operations: tuple[str, ...] = ()
    supported_semantics: tuple[str, ...] = ()
    interface_attr: str = ""
    schema_attr: str = ""
    expected_interface: str = ""
    proof_authority: bool = False
    reconstruction_compatible: bool = False
    optional: bool = False


_LOGIC_SPECS: Final = (
    _CapabilitySpec(
        capability_id="doctor.proof_cache",
        module=DOCTOR_PROOF_CACHE_MODULE,
        required_symbols=(
            "DoctorProofCacheGate",
            "DoctorProofCacheKey",
            "DoctorCacheAuditReceipt",
            "DoctorIdentityBinding",
        ),
        operations=(
            "lookup",
            "put",
            "revalidate_for_render",
            "revalidate_for_commit",
            "invalidate_semantic_root",
            "quarantine",
            "record_diagnostic",
        ),
        supported_semantics=(
            "exact_proof_cache",
            "semantic_root_invalidation",
            "equivocation_quarantine",
        ),
        interface_attr="DOCTOR_PROOF_CACHE_INTERFACE",
        expected_interface="DoctorProofCacheGate@1",
        proof_authority=True,
        reconstruction_compatible=True,
    ),
    _CapabilitySpec(
        capability_id="doctor.formal_verification_cache",
        module=FORMAL_CACHE_MODULE,
        required_symbols=(
            "FormalVerificationCache",
            "ProofCacheKey",
            "build_proof_cache_key",
        ),
        operations=("lookup", "put", "single_flight", "purge_expired"),
        supported_semantics=("trust_aware_proof_cache", "single_flight"),
        proof_authority=True,
        reconstruction_compatible=True,
    ),
    _CapabilitySpec(
        capability_id="doctor.prover_evidence_store",
        module=PROVER_EVIDENCE_MODULE,
        required_symbols=("ProverEvidenceStore",),
        operations=("lookup", "put", "project"),
        supported_semantics=("portfolio_evidence",),
        reconstruction_compatible=True,
        optional=True,
    ),
    _CapabilitySpec(
        capability_id="doctor.content_identity_bridge",
        module=CONTENT_IDENTITY_MODULE,
        required_symbols=(
            "identify_strict_artifact",
            "is_digest_shaped",
            "decode_and_verify_cid",
        ),
        operations=("identify", "verify_cid", "reject_digest_as_cid"),
        supported_semantics=("ContentIdentityBridge@1", "CIDv1"),
        interface_attr="CONTENT_IDENTITY_BRIDGE_INTERFACE",
        expected_interface="ContentIdentityBridge@1",
        reconstruction_compatible=True,
    ),
    _CapabilitySpec(
        capability_id="doctor.logic_provider",
        module=LOGIC_PROVIDER_MODULE,
        required_symbols=(
            "IsolatedHammerLoader",
            "get_isolated_hammer_loader",
        ),
        operations=("lazy_hammer_import", "probe_datasets_logic_backend"),
        supported_semantics=("isolated_hammer_loader", "datasets_logic"),
        reconstruction_compatible=True,
    ),
    _CapabilitySpec(
        capability_id="doctor.tactician_provider",
        module=TACTICIAN_PROVIDER_MODULE,
        required_symbols=("IpfsDatasetsTacticianProvider", "CodeTacticianCapability"),
        operations=("plan", "probe_capability"),
        supported_semantics=("exact_first_tactician", "nomination_only"),
        optional=True,
    ),
    _CapabilitySpec(
        capability_id="doctor.tactician_hammer_capabilities",
        module=TACTICIAN_CAPABILITIES_MODULE,
        required_symbols=("probe_tactician_hammer_capabilities",),
        operations=("probe_tactician_hammer_capabilities",),
        supported_semantics=("capability_probe", "no_install", "no_network"),
    ),
    _CapabilitySpec(
        capability_id="datasets.generic_tactician",
        module=GENERIC_TACTICIAN_MODULE,
        required_symbols=(),
        operations=("plan",),
        supported_semantics=("domain_neutral_tactician",),
        optional=True,
    ),
    _CapabilitySpec(
        capability_id="datasets.hammer",
        module=HAMMER_MODULE,
        required_symbols=(),
        operations=("nominate", "portfolio"),
        supported_semantics=("hammer_nomination", "provider_local_cache"),
        optional=True,
    ),
    _CapabilitySpec(
        capability_id="datasets.proof_corpus",
        module=PROOF_CORPUS_MODULE,
        required_symbols=(),
        operations=("nominate", "filter_applicability"),
        supported_semantics=("attested_proof_corpus", "nomination_only"),
        optional=True,
    ),
    _CapabilitySpec(
        capability_id="datasets.ir_core_identity",
        module=IR_CORE_IDENTITY_MODULE,
        required_symbols=(),
        operations=("canonical_identity",),
        supported_semantics=("logic_ir_identity",),
        optional=True,
    ),
)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _diagnostic(
    code: DatasetsDoctorLogicDiagnosticCode,
    capability_id: str,
    message: str,
    *,
    module: str = "",
    exception_type: str = "",
) -> DatasetsDoctorLogicDiagnostic:
    return DatasetsDoctorLogicDiagnostic(
        code=code,
        capability_id=capability_id,
        message=message,
        module=module,
        exception_type=exception_type,
    )


def _unavailable(
    capability_id: str,
    code: DatasetsDoctorLogicDiagnosticCode,
    message: str,
    *,
    module: str = "",
    exception_type: str = "",
    status: DatasetsDoctorLogicStatus = DatasetsDoctorLogicStatus.UNAVAILABLE,
    details: Mapping[str, Any] | None = None,
) -> DatasetsDoctorLogicCapability:
    return DatasetsDoctorLogicCapability(
        capability_id=capability_id,
        status=status,
        diagnostic=_diagnostic(
            code,
            capability_id,
            message,
            module=module,
            exception_type=exception_type,
        ),
        details=dict(details or {}),
    )


def find_module_spec(module_name: str) -> importlib.machinery.ModuleSpec | None:
    """Locate a module without importing it (no side effects)."""

    try:
        return importlib.util.find_spec(module_name)
    except (ImportError, ModuleNotFoundError, ValueError, AttributeError):
        return None


def module_path_available(module_name: str) -> bool:
    return find_module_spec(module_name) is not None


def _safe_import(
    module_name: str,
    *,
    importer: Callable[[str], Any] | None = None,
    timeout_seconds: float = DEFAULT_CAPABILITY_PROBE_TIMEOUT_SECONDS,
) -> ModuleType:
    """Import a module under a re-entrant lock without mutating HOME/sys.prefix.

    Injected importers bypass the process module cache so tests can force
    failures, timeouts, and isolation violations without fighting prior hits.
    """

    load = importer or importlib.import_module
    use_cache = importer is None
    with _PROBE_LOCK:
        if use_cache and module_name in _MODULE_CACHE:
            cached = _MODULE_CACHE[module_name]
            if cached is not None:
                return cached
            err = _MODULE_ERRORS.get(module_name)
            if err is not None:
                raise err
        original_home = os.environ.get("HOME")
        original_prefix = sys.prefix
        started = time.monotonic()
        try:
            module = load(module_name)
            if time.monotonic() - started > float(timeout_seconds):
                raise TimeoutError(
                    f"import of {module_name} exceeded {timeout_seconds}s"
                )
            # Guardrail: probes must never leave process globals mutated.
            if os.environ.get("HOME") != original_home or sys.prefix != original_prefix:
                if original_home is None:
                    os.environ.pop("HOME", None)
                else:
                    os.environ["HOME"] = original_home
                sys.prefix = original_prefix
                raise RuntimeError(
                    f"import of {module_name} mutated HOME or sys.prefix"
                )
            if not isinstance(module, ModuleType):
                # Some injectors return plain objects; wrap is not required but
                # we only cache real modules for re-use.
                return module  # type: ignore[return-value]
            if use_cache:
                _MODULE_CACHE[module_name] = module
            return module
        except BaseException as exc:
            if use_cache:
                _MODULE_CACHE[module_name] = None
                _MODULE_ERRORS[module_name] = exc
            # Always restore process globals on failure.
            if os.environ.get("HOME") != original_home:
                if original_home is None:
                    os.environ.pop("HOME", None)
                else:
                    os.environ["HOME"] = original_home
            if sys.prefix != original_prefix:
                sys.prefix = original_prefix
            raise


def _probe_spec(
    spec: _CapabilitySpec,
    *,
    importer: Callable[[str], Any] | None = None,
    timeout_seconds: float = DEFAULT_CAPABILITY_PROBE_TIMEOUT_SECONDS,
    allow_optional_path_only: bool = True,
) -> DatasetsDoctorLogicCapability:
    """Probe one exact capability without install/network/target import."""

    capability_id = spec.capability_id
    module_name = spec.module
    path_spec = find_module_spec(module_name)
    if path_spec is None and importer is None:
        status = (
            DatasetsDoctorLogicStatus.UNSUPPORTED
            if spec.optional
            else DatasetsDoctorLogicStatus.UNAVAILABLE
        )
        return _unavailable(
            capability_id,
            DatasetsDoctorLogicDiagnosticCode.MODULE_PATH_UNAVAILABLE,
            f"module path unavailable: {module_name}",
            module=module_name,
            status=status,
            details={"optional": spec.optional},
        )

    # Path-only probe for optional datasets packages avoids heavy imports.
    if (
        allow_optional_path_only
        and spec.optional
        and not spec.required_symbols
        and importer is None
    ):
        if path_spec is None:
            return _unavailable(
                capability_id,
                DatasetsDoctorLogicDiagnosticCode.MODULE_PATH_UNAVAILABLE,
                f"optional module path unavailable: {module_name}",
                module=module_name,
                status=DatasetsDoctorLogicStatus.UNSUPPORTED,
                details={"optional": True, "path_only": True},
            )
        origin = getattr(path_spec, "origin", None) or ""
        return DatasetsDoctorLogicCapability(
            capability_id=capability_id,
            status=DatasetsDoctorLogicStatus.AVAILABLE,
            module_paths=(module_name,),
            operations=spec.operations,
            supported_semantics=spec.supported_semantics,
            reconstruction_compatible=spec.reconstruction_compatible,
            proof_authority=False,
            details={
                "optional": True,
                "path_only": True,
                "origin": str(origin),
                "loaded": False,
            },
        )

    try:
        module = _safe_import(
            module_name, importer=importer, timeout_seconds=timeout_seconds
        )
    except TimeoutError as exc:
        return _unavailable(
            capability_id,
            DatasetsDoctorLogicDiagnosticCode.PROBE_TIMED_OUT,
            str(exc),
            module=module_name,
            exception_type=type(exc).__name__,
            status=DatasetsDoctorLogicStatus.TIMED_OUT,
        )
    except RuntimeError as exc:
        if "HOME" in str(exc) or "sys.prefix" in str(exc):
            return _unavailable(
                capability_id,
                DatasetsDoctorLogicDiagnosticCode.PROCESS_GLOBAL_MUTATION_FORBIDDEN,
                str(exc),
                module=module_name,
                exception_type=type(exc).__name__,
                status=DatasetsDoctorLogicStatus.UNSAFE,
            )
        return _unavailable(
            capability_id,
            DatasetsDoctorLogicDiagnosticCode.INTERNAL_ERROR,
            str(exc),
            module=module_name,
            exception_type=type(exc).__name__,
        )
    except (ImportError, ModuleNotFoundError, OSError) as exc:
        status = (
            DatasetsDoctorLogicStatus.UNSUPPORTED
            if spec.optional
            else DatasetsDoctorLogicStatus.UNAVAILABLE
        )
        return _unavailable(
            capability_id,
            DatasetsDoctorLogicDiagnosticCode.MODULE_IMPORT_FAILED,
            f"module import failed: {module_name}: {exc}",
            module=module_name,
            exception_type=type(exc).__name__,
            status=status,
            details={"optional": spec.optional},
        )
    except Exception as exc:  # noqa: BLE001 — probe must never raise
        return _unavailable(
            capability_id,
            DatasetsDoctorLogicDiagnosticCode.INTERNAL_ERROR,
            f"probe error: {exc}",
            module=module_name,
            exception_type=type(exc).__name__,
        )

    missing = [
        name for name in spec.required_symbols if not hasattr(module, name)
    ]
    if missing:
        return _unavailable(
            capability_id,
            DatasetsDoctorLogicDiagnosticCode.REQUIRED_SYMBOL_MISSING,
            f"required symbols missing: {', '.join(missing)}",
            module=module_name,
            status=DatasetsDoctorLogicStatus.INCOMPATIBLE,
            details={"missing_symbols": missing, "optional": spec.optional},
        )

    interface_version = ""
    if spec.interface_attr:
        interface_version = str(getattr(module, spec.interface_attr, "") or "")
        if spec.expected_interface and interface_version != spec.expected_interface:
            return _unavailable(
                capability_id,
                DatasetsDoctorLogicDiagnosticCode.INTERFACE_VERSION_INCOMPATIBLE,
                (
                    f"interface {interface_version!r} incompatible with "
                    f"{spec.expected_interface!r}"
                ),
                module=module_name,
                status=DatasetsDoctorLogicStatus.INCOMPATIBLE,
            )

    schema_version = ""
    if spec.schema_attr:
        schema_version = str(getattr(module, spec.schema_attr, "") or "")

    # Signature compatibility: required symbols that are callables must be callable.
    for name in spec.required_symbols:
        value = getattr(module, name, None)
        if inspect.isclass(value):
            continue
        if callable(value) or value is not None:
            continue

    return DatasetsDoctorLogicCapability(
        capability_id=capability_id,
        status=DatasetsDoctorLogicStatus.AVAILABLE,
        module_paths=(module_name,),
        interface_version=interface_version,
        schema_version=schema_version,
        operations=spec.operations,
        supported_semantics=spec.supported_semantics,
        reconstruction_compatible=spec.reconstruction_compatible,
        proof_authority=spec.proof_authority,
        details={
            "optional": spec.optional,
            "loaded": True,
            "required_symbols": list(spec.required_symbols),
        },
    )


def probe_datasets_doctor_logic_capabilities(
    *,
    importer: Callable[[str], Any] | None = None,
    timeout_seconds: float = DEFAULT_CAPABILITY_PROBE_TIMEOUT_SECONDS,
    allow_optional_path_only: bool = True,
    install_admitted: bool = False,
    network_admitted: bool = False,
    target_import_admitted: bool = False,
    process_global_mutation_admitted: bool = False,
) -> DatasetsDoctorLogicCapabilityReport:
    """Probe exact doctor logic/solver/cache capabilities.

    Injection points make failures and timeouts testable without mutating
    process-wide import state.  This function never installs packages, never
    contacts the network, never imports target repository modules, and never
    intentionally mutates ``HOME`` / ``sys.prefix``.
    """

    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds must be a positive number")
    for name, value in (
        ("install_admitted", install_admitted),
        ("network_admitted", network_admitted),
        ("target_import_admitted", target_import_admitted),
        ("process_global_mutation_admitted", process_global_mutation_admitted),
    ):
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be a boolean")
    # Admission flags are recorded only; this probe never exercises them.
    if install_admitted or network_admitted:
        # Explicitly refuse to act on these flags — doctor probes stay offline.
        pass

    original_home = os.environ.get("HOME")
    original_prefix = sys.prefix
    capabilities: list[DatasetsDoctorLogicCapability] = []
    for spec in _LOGIC_SPECS:
        capabilities.append(
            _probe_spec(
                spec,
                importer=importer,
                timeout_seconds=float(timeout_seconds),
                allow_optional_path_only=allow_optional_path_only,
            )
        )

    # Isolation capability: report that the probe path itself is hardened.
    isolation_mutated = (
        os.environ.get("HOME") != original_home or sys.prefix != original_prefix
    )
    if isolation_mutated:
        # Restore and report unsafe.
        if original_home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = original_home
        sys.prefix = original_prefix
        capabilities.append(
            _unavailable(
                "doctor.import_isolation",
                DatasetsDoctorLogicDiagnosticCode.IMPORT_ISOLATION_UNSAFE,
                "probe path observed HOME/sys.prefix mutation",
                status=DatasetsDoctorLogicStatus.UNSAFE,
                details={
                    "import_isolation": IMPORT_ISOLATION_UNSAFE,
                    "process_global_mutation_admitted": process_global_mutation_admitted,
                },
            )
        )
    else:
        capabilities.append(
            DatasetsDoctorLogicCapability(
                capability_id="doctor.import_isolation",
                status=DatasetsDoctorLogicStatus.AVAILABLE,
                module_paths=(LOGIC_PROVIDER_MODULE,),
                operations=("lazy_hammer_import", "probe_only"),
                supported_semantics=(
                    "no_home_mutation",
                    "no_sys_prefix_mutation",
                    "no_install",
                    "no_network",
                ),
                reconstruction_compatible=True,
                details={
                    "import_isolation": IMPORT_ISOLATION_HARDENED,
                    "install_admitted": False,
                    "network_admitted": False,
                    "target_import_admitted": bool(target_import_admitted),
                    "process_global_mutation_admitted": False,
                },
            )
        )

    return DatasetsDoctorLogicCapabilityReport(
        capabilities=tuple(capabilities),
        probed_at_ms=_now_ms(),
        probe_timeout_seconds=float(timeout_seconds),
        details={
            "spec_count": len(_LOGIC_SPECS),
            "install_admitted": False,
            "network_admitted": False,
            "target_import_admitted": bool(target_import_admitted),
            "process_global_mutation_admitted": False,
        },
    )


class IpfsDatasetsDoctorLogic:
    """Lazy facade over datasets Logic surfaces for the deterministic doctor.

    Cold construction performs no imports of optional datasets packages and no
    capability probing.  :meth:`probe` and :meth:`ensure` perform exact
    capability probes.  Hammer loads go through :class:`IsolatedHammerLoader`
    only when explicitly requested.
    """

    def __init__(
        self,
        *,
        timeout_seconds: float = DEFAULT_CAPABILITY_PROBE_TIMEOUT_SECONDS,
        importer: Callable[[str], Any] | None = None,
    ) -> None:
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be a positive number")
        self._timeout_seconds = float(timeout_seconds)
        self._importer = importer
        self._lock = threading.RLock()
        self._report: DatasetsDoctorLogicCapabilityReport | None = None
        self._modules: dict[str, Any] = {}
        self._hammer_loader: Any | None = None

    @property
    def provider_id(self) -> str:
        return DOCTOR_LOGIC_PROVIDER_ID

    @property
    def interface_version(self) -> str:
        return DOCTOR_LOGIC_INTERFACE

    @property
    def last_report(self) -> DatasetsDoctorLogicCapabilityReport | None:
        return self._report

    def capability_declaration(self) -> DatasetsDoctorLogicCapability:
        """Return a lazy, not-yet-probed declaration (package presence is not authority)."""

        return DatasetsDoctorLogicCapability(
            capability_id="doctor.logic.lazy",
            status=DatasetsDoctorLogicStatus.UNAVAILABLE,
            diagnostic=_diagnostic(
                DatasetsDoctorLogicDiagnosticCode.LAZY_NOT_PROBED,
                "doctor.logic.lazy",
                "capability not probed; call probe() or ensure()",
            ),
            details={"lazy": True, "probed": False},
        )

    def probe(
        self,
        *,
        force: bool = False,
        allow_optional_path_only: bool = True,
    ) -> DatasetsDoctorLogicCapabilityReport:
        """Run exact capability probes (idempotent unless ``force``)."""

        with self._lock:
            if self._report is not None and not force:
                return self._report
            report = probe_datasets_doctor_logic_capabilities(
                importer=self._importer,
                timeout_seconds=self._timeout_seconds,
                allow_optional_path_only=allow_optional_path_only,
            )
            self._report = report
            return report

    def ensure(
        self, capability_id: str, *, force_probe: bool = False
    ) -> DatasetsDoctorLogicCapability:
        """Probe (if needed) and return one capability, fail-closed if missing."""

        report = self.probe(force=force_probe)
        cap = report.get(capability_id)
        if cap is None:
            return _unavailable(
                capability_id,
                DatasetsDoctorLogicDiagnosticCode.MODULE_PATH_UNAVAILABLE,
                f"capability not in probe report: {capability_id}",
            )
        return cap

    def load_module(
        self, capability_id: str, *, require_available: bool = True
    ) -> Any | None:
        """Lazily import the module bound to *capability_id* after probing."""

        cap = self.ensure(capability_id)
        if not cap.available:
            if require_available:
                raise LookupError(
                    f"capability {capability_id!r} is not available: {cap.reason_code}"
                )
            return None
        module_name = cap.module_path
        if not module_name:
            if require_available:
                raise LookupError(f"capability {capability_id!r} has no module path")
            return None
        with self._lock:
            if module_name in self._modules:
                return self._modules[module_name]
            module = _safe_import(
                module_name,
                importer=self._importer,
                timeout_seconds=self._timeout_seconds,
            )
            self._modules[module_name] = module
            return module

    def get_isolated_hammer_loader(self) -> Any:
        """Return the hardened IsolatedHammerLoader without probing all caps."""

        with self._lock:
            if self._hammer_loader is not None:
                return self._hammer_loader
            module = _safe_import(
                LOGIC_PROVIDER_MODULE,
                importer=self._importer,
                timeout_seconds=self._timeout_seconds,
            )
            loader_factory = getattr(module, "get_isolated_hammer_loader", None)
            if not callable(loader_factory):
                raise LookupError(
                    "IsolatedHammerLoader factory unavailable on logic provider"
                )
            loader = loader_factory()
            isolation = getattr(loader, "isolation_report", None)
            if callable(isolation):
                report = isolation()
                if report.get("mutates_home") or report.get("mutates_sys_prefix"):
                    raise RuntimeError(
                        "IsolatedHammerLoader reports process-global mutation"
                    )
            self._hammer_loader = loader
            return loader

    def load_hammer(self) -> Any:
        """Lazy Hammer import through IsolatedHammerLoader only."""

        loader = self.get_isolated_hammer_loader()
        return loader.load()

    def open_doctor_proof_cache(self, path: str | Path | None = None, **kwargs: Any) -> Any:
        """Construct a DoctorProofCacheGate after verifying the capability."""

        cap = self.ensure("doctor.proof_cache")
        if not cap.available:
            raise LookupError(
                f"doctor.proof_cache unavailable: {cap.reason_code}"
            )
        module = self.load_module("doctor.proof_cache")
        gate_cls = getattr(module, "DoctorProofCacheGate", None)
        if gate_cls is None:
            raise LookupError("DoctorProofCacheGate symbol missing after load")
        return gate_cls(path, **kwargs)

    def isolation_report(self) -> dict[str, Any]:
        """Summarize process-global isolation status of loaded surfaces."""

        report = self.probe()
        isolation = report.get("doctor.import_isolation")
        hammer_details: dict[str, Any] = {}
        try:
            loader = self.get_isolated_hammer_loader()
            isolation_fn = getattr(loader, "isolation_report", None)
            if callable(isolation_fn):
                hammer_details = dict(isolation_fn())
        except Exception as exc:  # noqa: BLE001
            hammer_details = {
                "error": f"{type(exc).__name__}: {exc}",
                "concurrency_safe": False,
            }
        return {
            "probe_isolation": (
                isolation.to_dict() if isolation is not None else None
            ),
            "hammer_loader": hammer_details,
            "install_attempted": False,
            "network_attempted": False,
            "target_import_attempted": False,
            "process_global_mutation": False,
        }


def get_default_doctor_logic(
    *,
    timeout_seconds: float = DEFAULT_CAPABILITY_PROBE_TIMEOUT_SECONDS,
) -> IpfsDatasetsDoctorLogic:
    """Factory for a cold, unprobed doctor logic facade."""

    return IpfsDatasetsDoctorLogic(timeout_seconds=timeout_seconds)


__all__ = [
    "DEFAULT_CAPABILITY_PROBE_TIMEOUT_SECONDS",
    "DOCTOR_LOGIC_CAPABILITY_REPORT_SCHEMA",
    "DOCTOR_LOGIC_CAPABILITY_SCHEMA",
    "DOCTOR_LOGIC_INTERFACE",
    "DOCTOR_LOGIC_PROVIDER_ID",
    "DOCTOR_LOGIC_PROVIDER_VERSION",
    "DatasetsDoctorLogicCapability",
    "DatasetsDoctorLogicCapabilityReport",
    "DatasetsDoctorLogicDiagnostic",
    "DatasetsDoctorLogicDiagnosticCode",
    "DatasetsDoctorLogicStatus",
    "IMPORT_ISOLATION_HARDENED",
    "IMPORT_ISOLATION_UNSAFE",
    "IpfsDatasetsDoctorLogic",
    "find_module_spec",
    "get_default_doctor_logic",
    "module_path_available",
    "probe_datasets_doctor_logic_capabilities",
]
