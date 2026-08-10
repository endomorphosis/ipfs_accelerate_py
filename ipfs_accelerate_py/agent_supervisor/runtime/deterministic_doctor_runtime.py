"""Lazy production composition root for the deterministic Doctor.

The control service intentionally owns transport-neutral request/policy
contracts.  This higher layer owns the filesystem-facing work needed by a
normal ``inspect --checkout-root`` invocation:

* resolve one exact checkout against an explicit allowlist;
* build the existing canonical planning-analysis snapshot;
* enumerate policy-admitted sources and configured submodules without
  importing target code;
* compile repository diagnostics and bridge them to the deterministic Doctor
  contracts; and
* expose every later Doctor stage through a lazy, capability-reporting backend
  factory.

Import, construction, and :meth:`discovery` are cold.  In particular they do
not construct ``PlanningAnalysisFactory`` (which owns an index directory),
start a provider/process, open a database, or inspect a checkout.  Stage
imports and factories run only after an operation requests that stage.
"""

from __future__ import annotations

import hashlib
import importlib
import os
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ..analysis.deterministic_doctor_contracts import (
    DeterministicDoctorRunReceipt,
    DoctorEvidenceSnapshot,
    DoctorOperation,
    DoctorRepairDisposition,
)
from ..control.deterministic_doctor_service import (
    DeterministicDoctorService,
    DoctorOperationRequest,
    DoctorOperationResult,
    DoctorServiceCapabilityCode,
    DoctorServiceSafetyError,
    DoctorStageBackends,
    create_deterministic_doctor_service,
)
from ..proof.formal_verification_contracts import content_identity
from ..validation.deterministic_doctor_policy import DeterministicDoctorPolicy

DETERMINISTIC_DOCTOR_BACKEND_FACTORY_INTERFACE: Final[str] = (
    "DeterministicDoctorBackendFactory@1"
)
DETERMINISTIC_DOCTOR_RUNTIME_INTERFACE: Final[str] = "DeterministicDoctorRuntime@1"
DETERMINISTIC_DOCTOR_RUNTIME_DISCOVERY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor/runtime-discovery@1"
)
DETERMINISTIC_DOCTOR_RUNTIME_REPORT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor/runtime-report@1"
)
DETERMINISTIC_DOCTOR_EVIDENCE_BUNDLE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor/runtime-evidence@1"
)

# The deterministic snapshot contract intentionally bounds direct blob
# references.  The complete path ledger remains in PlanningAnalysisView; this
# is only the bounded parser batch projected into DoctorEvidenceSnapshot.
MAX_DIAGNOSTIC_SOURCE_PATHS: Final[int] = 256
MAX_DIAGNOSTIC_SOURCE_BYTES: Final[int] = 16 * 1024 * 1024
MAX_DIAGNOSTIC_TOTAL_BYTES: Final[int] = 256 * 1024 * 1024

_MODEL_MODULE_ROOTS: Final[frozenset[str]] = frozenset(
    {
        "anthropic",
        "llm_router",
        "openai",
        "torch",
        "transformers",
    }
)
_NETWORK_MODULE_ROOTS: Final[frozenset[str]] = frozenset(
    {"aiohttp", "httpx", "requests"}
)


class DeterministicDoctorRuntimeError(RuntimeError):
    """Base runtime composition or exact-evidence failure."""

    def __init__(self, reason_code: str, message: str = "") -> None:
        self.reason_code = str(reason_code or "runtime_error")
        super().__init__(str(message or reason_code))


class DeterministicDoctorRuntimeSafetyError(
    DoctorServiceSafetyError, DeterministicDoctorRuntimeError
):
    """A model/network/target-import route was attempted in deterministic mode."""

    def __init__(self, reason_code: str, message: str = "") -> None:
        self.reason_code = str(reason_code or "runtime_safety_error")
        DoctorServiceSafetyError.__init__(self, str(message or reason_code))


class DoctorRuntimeStageUnavailable(DeterministicDoctorRuntimeError):
    """One requested lazy stage is unavailable with an actionable remedy."""

    def __init__(
        self,
        stage: DoctorRuntimeStage | str,
        reason_code: str,
        remediation: str,
        *,
        cause: BaseException | None = None,
    ) -> None:
        self.stage = (
            stage.value if isinstance(stage, DoctorRuntimeStage) else str(stage)
        )
        self.remediation = str(remediation)
        message = (
            f"stage {self.stage!r} is unavailable ({reason_code}); "
            f"{self.remediation}"
        )
        super().__init__(reason_code, message)
        self.__cause__ = cause


class DoctorRuntimeStage(str, Enum):
    """Closed lazy production pipeline."""

    EVIDENCE = "evidence"
    DIAGNOSE = "diagnose"
    RETRIEVE = "retrieve"
    TACTICIAN = "tactician"
    PROOF = "proof"
    SYNTHESIS_PREVIEW = "synthesis_preview"
    IMPACT = "impact"
    TRANSACTION = "transaction"
    FIXED_POINT = "fixed_point"


_STAGE_ORDER: Final[tuple[DoctorRuntimeStage, ...]] = tuple(DoctorRuntimeStage)
_STAGE_INTERFACES: Final[Mapping[DoctorRuntimeStage, str]] = MappingProxyType(
    {
        DoctorRuntimeStage.EVIDENCE: "PlanningAnalysisFactory@1",
        DoctorRuntimeStage.DIAGNOSE: "DoctorRepositoryDiagnostics@1",
        DoctorRuntimeStage.RETRIEVE: "DoctorRepairCandidateRetriever@1",
        DoctorRuntimeStage.TACTICIAN: "DeterministicDoctorTactician@1",
        DoctorRuntimeStage.PROOF: "DeterministicDoctorHammer@1",
        DoctorRuntimeStage.SYNTHESIS_PREVIEW: "DeterministicDoctorSynthesizer@1",
        DoctorRuntimeStage.IMPACT: "DeterministicDoctorImpact@1",
        DoctorRuntimeStage.TRANSACTION: "DeterministicDoctorTransaction@1",
        DoctorRuntimeStage.FIXED_POINT: "DeterministicDoctorFixedPoint@1",
    }
)
_STAGE_REMEDIATIONS: Final[Mapping[DoctorRuntimeStage, str]] = MappingProxyType(
    {
        DoctorRuntimeStage.EVIDENCE: (
            "install the repository-analysis package and bind an exact checkout"
        ),
        DoctorRuntimeStage.DIAGNOSE: (
            "install the inert AST adapters or inject a diagnose stage factory"
        ),
        DoctorRuntimeStage.RETRIEVE: (
            "bind deterministic candidate signals or inject a retrieval factory"
        ),
        DoctorRuntimeStage.TACTICIAN: (
            "bind the local deterministic tactician; model routes are forbidden"
        ),
        DoctorRuntimeStage.PROOF: (
            "install an approved digest-bound prover/toolchain or remain report-only"
        ),
        DoctorRuntimeStage.SYNTHESIS_PREVIEW: (
            "bind an admitted operator proposal and proof before synthesis preview"
        ),
        DoctorRuntimeStage.IMPACT: (
            "bind current program-graph/consumer evidence before impact closure"
        ),
        DoctorRuntimeStage.TRANSACTION: (
            "bind a real sandbox applicator and a control-plane permit/effect adapter"
        ),
        DoctorRuntimeStage.FIXED_POINT: (
            "bind transaction output and independently produced fixed-point evidence"
        ),
    }
)

# Production composition (DCR-050) requires these stage backends to be real
# callables, never empty slots or deferred placeholders.
MANDATORY_PRODUCTION_BACKENDS: Final[tuple[str, ...]] = (
    "diagnose",
    "plan",
    "retrieve",
    "tactician",
    "proof",
    "transaction",
)

# Lazy stages that may remain deferred until typed inputs arrive.  Deferred
# production stages report unavailable / abstain — never successful completion.
OPTIONAL_DEFERRED_BACKENDS: Final[tuple[str, ...]] = (
    "synthesis",
    "impact",
    "fixed_point",
    "explain",
)


@dataclass(frozen=True)
class DoctorStageCapability:
    """Static/lazy capability state for one Doctor stage."""

    stage: DoctorRuntimeStage
    interface: str
    declared: bool = True
    loaded: bool = False
    available: bool | None = None
    reason_code: str = "not_probed"
    remediation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "interface": self.interface,
            "declared": self.declared,
            "loaded": self.loaded,
            "available": self.available,
            "reason_code": self.reason_code,
            "remediation": self.remediation,
        }


@dataclass(frozen=True)
class DoctorSourceInventoryEntry:
    """Body-free exact source enumeration entry."""

    path: str
    content_digest: str
    coverage_kind: str
    byte_count: int
    root_kind: str = "primary"
    git_object_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "content_digest": self.content_digest,
            "coverage_kind": self.coverage_kind,
            "byte_count": self.byte_count,
            "root_kind": self.root_kind,
            "git_object_id": self.git_object_id,
        }


@dataclass(frozen=True)
class DeterministicDoctorEvidenceBundle:
    """Exact checkout evidence plus the checked Doctor schema bridge."""

    checkout_root: str
    analysis_view: Any
    diagnostic_snapshot: Any
    snapshot: DoctorEvidenceSnapshot
    findings: tuple[Any, ...]
    source_inventory: tuple[DoctorSourceInventoryEntry, ...]
    diagnostic_source_paths: tuple[str, ...]
    submodule_closure: tuple[Mapping[str, Any], ...]
    bridge_id: str
    notes: tuple[str, ...] = ()

    @property
    def evidence_id(self) -> str:
        return content_identity(
            {
                "schema": DETERMINISTIC_DOCTOR_EVIDENCE_BUNDLE_SCHEMA,
                "checkout_root": self.checkout_root,
                "analysis_view_id": self.analysis_view.view_cid,
                "snapshot_id": self.snapshot.snapshot_id,
                "snapshot_content_id": self.snapshot.content_id,
                "source_inventory": [item.to_dict() for item in self.source_inventory],
                "submodule_closure": list(self.submodule_closure),
                "bridge_id": self.bridge_id,
                "notes": list(self.notes),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DETERMINISTIC_DOCTOR_EVIDENCE_BUNDLE_SCHEMA,
            "interface": DETERMINISTIC_DOCTOR_RUNTIME_INTERFACE,
            "evidence_id": self.evidence_id,
            "checkout_root": self.checkout_root,
            "analysis_view_id": self.analysis_view.view_cid,
            "analysis_completeness": self.analysis_view.completeness,
            "snapshot": self.snapshot.to_dict(),
            "finding_ids": [
                str(getattr(item, "finding_id", "") or getattr(item, "content_id", ""))
                for item in self.findings
            ],
            "source_inventory": [item.to_dict() for item in self.source_inventory],
            "diagnostic_source_paths": list(self.diagnostic_source_paths),
            "submodule_closure": [dict(item) for item in self.submodule_closure],
            "bridge_id": self.bridge_id,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class DeterministicDoctorRuntimeReport:
    """One service result enriched with body-free production stage evidence."""

    result: DoctorOperationResult
    evidence: DeterministicDoctorEvidenceBundle | None = None
    stage_receipts: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DETERMINISTIC_DOCTOR_RUNTIME_REPORT_SCHEMA,
            "interface": DETERMINISTIC_DOCTOR_RUNTIME_INTERFACE,
            "result": self.result.to_dict(),
            "evidence": self.evidence.to_dict() if self.evidence is not None else None,
            "stage_receipts": {
                str(key): dict(value)
                for key, value in sorted(self.stage_receipts.items())
            },
        }


def _import_symbol(module_name: str, symbol: str) -> Any:
    module = importlib.import_module(module_name)
    return getattr(module, symbol)


def _default_stage_loaders() -> dict[DoctorRuntimeStage, Callable[[], Any]]:
    """Return closures only; importing this module imports no stage provider."""

    return {
        DoctorRuntimeStage.EVIDENCE: lambda: _import_symbol(
            "ipfs_accelerate_py.agent_supervisor.analysis.planning_analysis_factory",
            "PlanningAnalysisFactory",
        ),
        DoctorRuntimeStage.DIAGNOSE: lambda: _import_symbol(
            "ipfs_accelerate_py.agent_supervisor.analysis.doctor_repository_diagnostics",
            "diagnose_repository",
        ),
        DoctorRuntimeStage.RETRIEVE: lambda: _import_symbol(
            "ipfs_accelerate_py.agent_supervisor.analysis.doctor_repair_candidate_retrieval",
            "DoctorRepairCandidateRetriever",
        ),
        DoctorRuntimeStage.TACTICIAN: lambda: _import_symbol(
            "ipfs_accelerate_py.agent_supervisor.planning.deterministic_doctor_tactician",
            "DeterministicDoctorTactician",
        ),
        DoctorRuntimeStage.PROOF: lambda: _import_symbol(
            "ipfs_accelerate_py.agent_supervisor.proof.deterministic_doctor_hammer",
            "DeterministicDoctorHammer",
        ),
        DoctorRuntimeStage.SYNTHESIS_PREVIEW: lambda: _import_symbol(
            "ipfs_accelerate_py.agent_supervisor.planning.deterministic_doctor_synthesis",
            "create_deterministic_doctor_synthesizer",
        ),
        DoctorRuntimeStage.IMPACT: lambda: _import_symbol(
            "ipfs_accelerate_py.agent_supervisor.analysis.deterministic_doctor_impact",
            "DeterministicDoctorImpactAnalyzer",
        ),
        DoctorRuntimeStage.TRANSACTION: lambda: _import_symbol(
            "ipfs_accelerate_py.agent_supervisor.planning.deterministic_doctor_transaction",
            "DeterministicDoctorTransaction",
        ),
        DoctorRuntimeStage.FIXED_POINT: lambda: _import_symbol(
            "ipfs_accelerate_py.agent_supervisor.validation.deterministic_doctor_fixed_point",
            "DeterministicDoctorFixedPointValidator",
        ),
    }


class DeterministicDoctorBackendFactory:
    """Thread-safe lazy stage registry with deterministic-route enforcement."""

    INTERFACE: ClassVar[str] = DETERMINISTIC_DOCTOR_BACKEND_FACTORY_INTERFACE

    def __init__(
        self,
        *,
        stage_factories: Mapping[
            DoctorRuntimeStage | str, Callable[[], Any] | Any
        ]
        | None = None,
        deterministic: bool = True,
    ) -> None:
        loaders = _default_stage_loaders()
        for raw_stage, loader in dict(stage_factories or {}).items():
            stage = (
                raw_stage
                if isinstance(raw_stage, DoctorRuntimeStage)
                else DoctorRuntimeStage(str(raw_stage))
            )
            loaders[stage] = loader if callable(loader) else lambda value=loader: value
        self._loaders = loaders
        self._instances: dict[DoctorRuntimeStage, Any] = {}
        self._failures: dict[DoctorRuntimeStage, DoctorRuntimeStageUnavailable] = {}
        self._deterministic = bool(deterministic)
        self._lock = threading.RLock()

    @staticmethod
    def discovery() -> dict[str, Any]:
        """Return the declared graph without importing or probing a stage."""

        return {
            "schema": DETERMINISTIC_DOCTOR_RUNTIME_DISCOVERY_SCHEMA,
            "interface": DETERMINISTIC_DOCTOR_BACKEND_FACTORY_INTERFACE,
            "runtime_interface": DETERMINISTIC_DOCTOR_RUNTIME_INTERFACE,
            "stages": [
                DoctorStageCapability(
                    stage=stage,
                    interface=_STAGE_INTERFACES[stage],
                    remediation=_STAGE_REMEDIATIONS[stage],
                ).to_dict()
                for stage in _STAGE_ORDER
            ],
            "deterministic": True,
            "model_routes_allowed": False,
            "network_routes_allowed": False,
            "providers_started": False,
            "processes_started": False,
            "database_opened": False,
        }

    def capabilities(self) -> tuple[DoctorStageCapability, ...]:
        """Return current state without causing a probe or load."""

        with self._lock:
            rows: list[DoctorStageCapability] = []
            for stage in _STAGE_ORDER:
                failure = self._failures.get(stage)
                loaded = stage in self._instances
                rows.append(
                    DoctorStageCapability(
                        stage=stage,
                        interface=_STAGE_INTERFACES[stage],
                        loaded=loaded,
                        available=False if failure else True if loaded else None,
                        reason_code=(
                            failure.reason_code
                            if failure is not None
                            else "available"
                            if loaded
                            else "not_probed"
                        ),
                        remediation=(
                            failure.remediation
                            if failure is not None
                            else _STAGE_REMEDIATIONS[stage]
                        ),
                    )
                )
            return tuple(rows)

    @property
    def loaded_stages(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(stage.value for stage in _STAGE_ORDER if stage in self._instances)

    def get(self, stage: DoctorRuntimeStage | str) -> Any:
        selected = (
            stage if isinstance(stage, DoctorRuntimeStage) else DoctorRuntimeStage(stage)
        )
        with self._lock:
            if selected in self._instances:
                return self._instances[selected]
            if selected in self._failures:
                raise self._failures[selected]
            loader = self._loaders.get(selected)
            if loader is None:
                failure = DoctorRuntimeStageUnavailable(
                    selected,
                    "stage_factory_missing",
                    _STAGE_REMEDIATIONS[selected],
                )
                self._failures[selected] = failure
                raise failure
            before_modules = frozenset(sys.modules)
            try:
                instance = loader()
                self._assert_safe_route(selected, instance, before_modules)
            except DeterministicDoctorRuntimeSafetyError:
                raise
            except Exception as exc:
                failure = DoctorRuntimeStageUnavailable(
                    selected,
                    "stage_dependency_unavailable",
                    _STAGE_REMEDIATIONS[selected],
                    cause=exc,
                )
                self._failures[selected] = failure
                raise failure from exc
            self._instances[selected] = instance
            return instance

    def _assert_safe_route(
        self,
        stage: DoctorRuntimeStage,
        instance: Any,
        before_modules: frozenset[str],
    ) -> None:
        if not self._deterministic:
            return
        if bool(getattr(instance, "uses_model", False)) or bool(
            getattr(instance, "model_route", False)
        ):
            raise DeterministicDoctorRuntimeSafetyError(
                "model_route_forbidden",
                f"stage {stage.value!r} declared a model route",
            )
        if bool(getattr(instance, "uses_network", False)) or bool(
            getattr(instance, "network_route", False)
        ):
            raise DeterministicDoctorRuntimeSafetyError(
                "network_route_forbidden",
                f"stage {stage.value!r} declared a network route",
            )
        added_roots = {
            name.split(".", 1)[0] for name in set(sys.modules).difference(before_modules)
        }
        model = sorted(added_roots & _MODEL_MODULE_ROOTS)
        network = sorted(added_roots & _NETWORK_MODULE_ROOTS)
        if model:
            raise DeterministicDoctorRuntimeSafetyError(
                "model_route_forbidden",
                f"stage {stage.value!r} loaded model modules: {model}",
            )
        if network:
            raise DeterministicDoctorRuntimeSafetyError(
                "network_route_forbidden",
                f"stage {stage.value!r} loaded network modules: {network}",
            )


def _canonical_checkout(
    checkout_root: str | os.PathLike[str],
    repository_allowlist: Sequence[str | os.PathLike[str]],
) -> Path:
    try:
        root = Path(checkout_root).expanduser().resolve(strict=True)
    except OSError as exc:
        raise DeterministicDoctorRuntimeError(
            "checkout_unavailable", f"checkout root is unavailable: {checkout_root}"
        ) from exc
    if not root.is_dir():
        raise DeterministicDoctorRuntimeError(
            "checkout_not_directory", "checkout root must be a directory"
        )
    allowed: set[Path] = set()
    for candidate in repository_allowlist:
        try:
            allowed.add(Path(candidate).expanduser().resolve(strict=True))
        except OSError as exc:
            raise DeterministicDoctorRuntimeError(
                "allowlist_root_unavailable",
                f"allowlisted repository root is unavailable: {candidate}",
            ) from exc
    if root not in allowed:
        raise DeterministicDoctorRuntimeError(
            "checkout_not_allowlisted",
            "checkout root is not one of the explicit exact repository roots",
        )
    return root


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _submodule_records(entries: Sequence[Any]) -> tuple[Mapping[str, Any], ...]:
    records: list[Mapping[str, Any]] = []

    def visit(entry: Any, prefix: str = "") -> None:
        local = str(getattr(entry, "path", "") or "")
        joined = str(PurePosixPath(prefix, local)) if prefix else local
        records.append(
            MappingProxyType(
                {
                    "path": joined,
                    "commit_id": str(getattr(entry, "commit_id", "") or ""),
                    "depth": int(getattr(entry, "depth", 0)),
                    "available": bool(getattr(entry, "available", False)),
                    "reason_code": str(
                        getattr(entry, "reason_code", "configured_submodule")
                    ),
                }
            )
        )
        for child in tuple(getattr(entry, "nested", ()) or ()):
            visit(child, joined)

    for root_entry in entries:
        visit(root_entry)
    return tuple(sorted(records, key=lambda item: str(item["path"])))


class DeterministicDoctorRuntime:
    """Lazy production deterministic-Doctor runtime for one exact checkout."""

    INTERFACE: ClassVar[str] = DETERMINISTIC_DOCTOR_RUNTIME_INTERFACE

    def __init__(
        self,
        *,
        checkout_root: str | os.PathLike[str],
        repository_allowlist: Sequence[str | os.PathLike[str]] | None = None,
        policy: DeterministicDoctorPolicy | Mapping[str, Any] | None = None,
        backend_factory: DeterministicDoctorBackendFactory | None = None,
        control_service: Any | None = None,
        receipt_store: Any | None = None,
        scope_policy: Any | None = None,
        index_root: str | os.PathLike[str] | None = None,
        deterministic: bool = True,
    ) -> None:
        allowlist = tuple(repository_allowlist or (checkout_root,))
        self.checkout_root = _canonical_checkout(checkout_root, allowlist)
        self.repository_allowlist = tuple(
            sorted(str(Path(item).expanduser().resolve(strict=True)) for item in allowlist)
        )
        self._policy = policy
        self._factory = backend_factory or DeterministicDoctorBackendFactory(
            deterministic=deterministic
        )
        self._control_service = control_service
        self._scope_policy = scope_policy
        self._index_root = str(Path(index_root).resolve(strict=False)) if index_root else None
        self._deterministic = bool(deterministic)
        self._analysis_factory: Any | None = None
        self._evidence: DeterministicDoctorEvidenceBundle | None = None
        self._stage_receipts: dict[str, Mapping[str, Any]] = {}
        self._lock = threading.RLock()
        # Production composition (DCR-050): mandatory backends are real stage
        # adapters bound at construction.  Optional later stages may defer with
        # typed unavailability — never empty slots or silent success.
        self._service: DeterministicDoctorService = create_deterministic_doctor_service(
            policy=policy,
            receipt_store=receipt_store,
            control_service=control_service,
            backends=DoctorStageBackends(
                diagnose=self._diagnose_backend,
                plan=self._plan_backend,
                retrieve=self._retrieve_backend,
                tactician=self._tactician_backend,
                proof=self._proof_backend,
                transaction=self._transaction_backend,
                synthesis=self._deferred_stage_backend(
                    DoctorRuntimeStage.SYNTHESIS_PREVIEW
                ),
                impact=self._deferred_stage_backend(DoctorRuntimeStage.IMPACT),
                fixed_point=self._deferred_stage_backend(DoctorRuntimeStage.FIXED_POINT),
            ),
        )
        self._composition_handles: Mapping[str, Any] | None = None

    @staticmethod
    def discovery() -> dict[str, Any]:
        """Cold static discovery; no checkout or provider is touched."""

        manifest = DeterministicDoctorBackendFactory.discovery()
        return {
            **manifest,
            "interface": DETERMINISTIC_DOCTOR_RUNTIME_INTERFACE,
            "backend_factory_interface": DETERMINISTIC_DOCTOR_BACKEND_FACTORY_INTERFACE,
            "service_interface": DeterministicDoctorService.INTERFACE,
        }

    @property
    def service(self) -> DeterministicDoctorService:
        return self._service

    @property
    def backend_factory(self) -> DeterministicDoctorBackendFactory:
        return self._factory

    @property
    def evidence(self) -> DeterministicDoctorEvidenceBundle | None:
        return self._evidence

    @property
    def composition_handles(self) -> Mapping[str, Any] | None:
        """Optional production composition handles attached by the factory."""

        return self._composition_handles

    def attach_composition_handles(self, handles: Mapping[str, Any]) -> None:
        """Attach body-free production composition handles (idempotent)."""

        if not isinstance(handles, Mapping):
            raise DeterministicDoctorRuntimeError(
                "invalid_composition_handles",
                "composition handles must be a mapping",
            )
        self._composition_handles = MappingProxyType(dict(handles))

    def mandatory_backends_bound(self) -> tuple[str, ...]:
        """Return mandatory backends that are non-empty and non-deferred."""

        available = set(self._service.backends_available)
        bound: list[str] = []
        for name in MANDATORY_PRODUCTION_BACKENDS:
            backend = getattr(self._service._backends, name, None)  # noqa: SLF001
            if backend is None or name not in available:
                continue
            if bool(getattr(backend, "doctor_deferred_backend", False)):
                continue
            bound.append(name)
        return tuple(bound)

    def assert_mandatory_backends_production_ready(self) -> None:
        """Fail closed when any mandatory backend is empty or deferred."""

        bound = set(self.mandatory_backends_bound())
        missing = [name for name in MANDATORY_PRODUCTION_BACKENDS if name not in bound]
        if missing:
            raise DeterministicDoctorRuntimeError(
                "mandatory_backend_unavailable",
                "mandatory production backends are empty or deferred: "
                + ", ".join(missing),
            )

    def capability_graph(self) -> dict[str, Any]:
        """Report current lazy state without loading an unrequested stage."""

        return {
            "interface": DETERMINISTIC_DOCTOR_BACKEND_FACTORY_INTERFACE,
            "stages": [item.to_dict() for item in self._factory.capabilities()],
            "loaded_stages": list(self._factory.loaded_stages),
            "mandatory_backends": list(MANDATORY_PRODUCTION_BACKENDS),
            "mandatory_backends_bound": list(self.mandatory_backends_bound()),
            "optional_deferred_backends": list(OPTIONAL_DEFERRED_BACKENDS),
            "providers_started": False,
            "network_routes_allowed": False,
            "model_routes_allowed": False,
        }

    def inspect(self, **kwargs: Any) -> DeterministicDoctorRuntimeReport:
        return self.execute(DoctorOperation.INSPECT.value, **kwargs)

    def plan(self, **kwargs: Any) -> DeterministicDoctorRuntimeReport:
        return self.execute(DoctorOperation.PLAN.value, **kwargs)

    def execute(
        self,
        request: DoctorOperationRequest | Mapping[str, Any] | str,
        **kwargs: Any,
    ) -> DeterministicDoctorRuntimeReport:
        """Execute through the shared control service with runtime-built evidence."""

        if isinstance(request, DoctorOperationRequest):
            payload = request.to_dict()
            payload.pop("schema", None)
            payload.pop("content_id", None)
            payload.update(kwargs)
        elif isinstance(request, Mapping):
            payload = {**dict(request), **kwargs}
        else:
            payload = {"operation": str(request), **kwargs}
        operation = str(payload.get("operation") or "")
        if self._deterministic and (
            payload.get("network_access")
            or payload.get("llm_router_invoked")
            or payload.get("remote_model_provider_invoked")
            or int(payload.get("model_invocation_count") or 0)
            or int(payload.get("provider_invocation_count") or 0)
        ):
            raise DeterministicDoctorRuntimeSafetyError(
                "deterministic_route_forbidden",
                "deterministic Doctor forbids model/provider/network routes",
            )

        # Report/plan operations can be invoked without caller-authored JSON.
        if operation in {
            DoctorOperation.INSPECT.value,
            DoctorOperation.EXPLAIN.value,
            DoctorOperation.PLAN.value,
            DoctorOperation.REPAIR.value,
        } and "snapshot" not in payload:
            evidence = self.build_evidence()
            payload["snapshot"] = evidence.snapshot.to_dict()
            payload.setdefault("roots", evidence.snapshot.roots.to_dict())
            payload.setdefault(
                "finding_ids",
                tuple(
                    str(
                        getattr(item, "finding_id", "")
                        or getattr(item, "content_id", "")
                    )
                    for item in evidence.findings
                ),
            )
        result = self._service.execute(payload)
        return DeterministicDoctorRuntimeReport(
            result=result,
            evidence=self._evidence,
            stage_receipts=MappingProxyType(dict(self._stage_receipts)),
        )

    def build_evidence(self, *, refresh: bool = False) -> DeterministicDoctorEvidenceBundle:
        """Build or return exact body-free evidence for the bound checkout."""

        with self._lock:
            if self._evidence is not None and not refresh:
                return self._evidence
            analysis_factory_class = self._factory.get(DoctorRuntimeStage.EVIDENCE)
            if self._analysis_factory is None:
                self._analysis_factory = analysis_factory_class(
                    repository_allowlist=self.repository_allowlist,
                    index_root=self._index_root,
                    scope_policy=self._scope_policy,
                    optional_providers={},
                    build_index=True,
                )
            view = self._analysis_factory.analyze(self.checkout_root)
            inventory = self._enumerate_sources(view)
            notes: list[str] = ["exact_checkout_stability_verified"]
            diagnostic_paths_list: list[str] = []
            diagnostic_bytes = 0
            bounded_sources = False
            for item in inventory:
                if item.coverage_kind not in {"semantic_ast", "structured_data"}:
                    continue
                if item.byte_count > MAX_DIAGNOSTIC_SOURCE_BYTES:
                    bounded_sources = True
                    continue
                if (
                    len(diagnostic_paths_list) >= MAX_DIAGNOSTIC_SOURCE_PATHS
                    or diagnostic_bytes + item.byte_count
                    > MAX_DIAGNOSTIC_TOTAL_BYTES
                ):
                    bounded_sources = True
                    continue
                diagnostic_paths_list.append(item.path)
                diagnostic_bytes += item.byte_count
            diagnostic_paths = tuple(diagnostic_paths_list)
            if bounded_sources:
                notes.append("diagnostic_source_bound_reached")
            diag_snapshot, snapshot, findings, bridge_id = self._compile_diagnostics(
                view, diagnostic_paths, inventory
            )
            closure = _submodule_records(view.submodule_closure)
            bundle = DeterministicDoctorEvidenceBundle(
                checkout_root=str(self.checkout_root),
                analysis_view=view,
                diagnostic_snapshot=diag_snapshot,
                snapshot=snapshot,
                findings=findings,
                source_inventory=inventory,
                diagnostic_source_paths=diagnostic_paths,
                submodule_closure=closure,
                bridge_id=bridge_id,
                notes=tuple(notes),
            )
            self._evidence = bundle
            self._stage_receipts[DoctorRuntimeStage.EVIDENCE.value] = {
                "status": "completed",
                "evidence_id": bundle.evidence_id,
                "analysis_view_id": view.view_cid,
                "source_count": len(inventory),
                "submodule_count": len(closure),
            }
            self._stage_receipts[DoctorRuntimeStage.DIAGNOSE.value] = {
                "status": "completed",
                "snapshot_id": snapshot.snapshot_id,
                "finding_count": len(findings),
                "bridge_id": bridge_id,
            }
            return bundle

    def _enumerate_sources(
        self, view: Any
    ) -> tuple[DoctorSourceInventoryEntry, ...]:
        from ..analysis.repository_snapshot import CoverageKind, EntryKind

        rows: dict[str, DoctorSourceInventoryEntry] = {}

        def add_snapshot(snapshot: Any, prefix: str, root_kind: str) -> None:
            for disposition in snapshot.dispositions:
                if disposition.kind is CoverageKind.EXCLUDED:
                    continue
                if disposition.entry_kind is not EntryKind.REGULAR:
                    continue
                relative = (
                    str(PurePosixPath(prefix, disposition.path))
                    if prefix
                    else disposition.path
                )
                candidate = self.checkout_root.joinpath(*PurePosixPath(relative).parts)
                try:
                    if candidate.is_symlink() or not candidate.is_file():
                        continue
                    payload = candidate.read_bytes()
                except OSError as exc:
                    raise DeterministicDoctorRuntimeError(
                        "source_became_unreadable",
                        f"admitted source became unreadable: {relative}",
                    ) from exc
                digest = _sha256(payload)
                if disposition.content_digest and digest != disposition.content_digest:
                    raise DeterministicDoctorRuntimeError(
                        "source_identity_mismatch",
                        f"admitted source changed after snapshot: {relative}",
                    )
                rows[relative] = DoctorSourceInventoryEntry(
                    path=relative,
                    content_digest=digest,
                    coverage_kind=disposition.kind.value,
                    byte_count=len(payload),
                    root_kind=root_kind,
                    git_object_id=disposition.git_object_id,
                )

        add_snapshot(view.sca_snapshot, "", "primary")

        # A primary snapshot records gitlinks but not child source ledgers.
        # Only recursively configured and materialized submodules are expanded.
        from ..analysis.repository_snapshot import build_repository_snapshot

        def add_submodule(entry: Any, prefix: str = "") -> None:
            local = str(getattr(entry, "path", "") or "")
            joined = str(PurePosixPath(prefix, local)) if prefix else local
            if bool(getattr(entry, "available", False)):
                child_root = self.checkout_root.joinpath(*PurePosixPath(joined).parts)
                try:
                    child_snapshot = build_repository_snapshot(
                        child_root,
                        scope_policy=self._analysis_factory.scope_policy,
                        allow_dirty_analysis=True,
                    )
                    add_snapshot(child_snapshot, joined, "submodule")
                except Exception as exc:
                    raise DeterministicDoctorRuntimeError(
                        "submodule_inventory_unavailable",
                        f"configured submodule could not be inventoried: {joined}",
                    ) from exc
            for child in tuple(getattr(entry, "nested", ()) or ()):
                add_submodule(child, joined)

        for root_entry in view.submodule_closure:
            add_submodule(root_entry)
        return tuple(rows[path] for path in sorted(rows))

    def _compile_diagnostics(
        self,
        view: Any,
        diagnostic_paths: Sequence[str],
        inventory: Sequence[DoctorSourceInventoryEntry],
    ) -> tuple[Any, DoctorEvidenceSnapshot, tuple[Any, ...], str]:
        diagnose = self._factory.get(DoctorRuntimeStage.DIAGNOSE)
        diagnostics_module = importlib.import_module(
            "ipfs_accelerate_py.agent_supervisor.analysis.doctor_repository_diagnostics"
        )
        roots = view.reasoning_snapshot.roots
        diag_roots = diagnostics_module.DoctorAuthorityRoots(
            repository_id=roots.repository_id,
            forest_id=roots.forest_id,
            tree_id=roots.tree_id,
            overlay_id=roots.overlay_id,
            file_root_id=view.sca_snapshot.snapshot_id,
            blob_root_id=content_identity(
                {
                    "snapshot_id": view.sca_snapshot.snapshot_id,
                    "paths": list(diagnostic_paths),
                }
            ),
            parser_id=roots.parser_root,
            config_id=roots.scope_policy_id,
            toolchain_id=roots.toolchain_root,
            policy_id=roots.policy_root,
            ast_index_id=roots.ast_root,
            symbol_index_id=roots.index_root,
            import_graph_id=roots.program_behavior_root,
            dependency_graph_id=roots.program_behavior_root,
            evidence_graph_id=view.view_cid,
            corpus_root_id=view.sca_snapshot.snapshot_id,
        )
        digest_by_path = {item.path: item.content_digest for item in inventory}
        byte_count_by_path = {item.path: item.byte_count for item in inventory}
        # Production composition (DCR-050): empty source bytes are unavailable,
        # not successful.  Load exact admitted bytes from the checkout inventory.
        source_units_list: list[Any] = []
        for path in diagnostic_paths:
            candidate = self.checkout_root.joinpath(*PurePosixPath(path).parts)
            try:
                if candidate.is_symlink() or not candidate.is_file():
                    raise DeterministicDoctorRuntimeError(
                        "source_became_unreadable",
                        f"admitted diagnostic source is not a regular file: {path}",
                    )
                payload = candidate.read_bytes()
            except OSError as exc:
                raise DeterministicDoctorRuntimeError(
                    "source_became_unreadable",
                    f"admitted diagnostic source became unreadable: {path}",
                ) from exc
            if not payload:
                # Empty bodies cannot establish production diagnostic evidence.
                raise DeterministicDoctorRuntimeError(
                    "empty_source_unavailable",
                    f"admitted diagnostic source has empty bytes: {path}",
                )
            digest = _sha256(payload)
            expected = digest_by_path.get(path, "")
            if expected and digest != expected:
                raise DeterministicDoctorRuntimeError(
                    "source_identity_mismatch",
                    f"admitted source changed after snapshot: {path}",
                )
            if byte_count_by_path.get(path, -1) not in {-1, len(payload)}:
                raise DeterministicDoctorRuntimeError(
                    "source_identity_mismatch",
                    f"admitted source size drifted after inventory: {path}",
                )
            source_units_list.append(
                diagnostics_module.DoctorSourceUnit(
                    path=path,
                    source_bytes=payload,
                    blob_identity=digest,
                )
            )
        source_units = tuple(source_units_list)
        diag_snapshot = diagnose(
            sources=source_units,
            repository_root=str(self.checkout_root),
            authority_roots=diag_roots,
            policy={
                "max_paths": max(1, len(source_units)),
                "max_source_bytes": MAX_DIAGNOSTIC_SOURCE_BYTES,
                "max_total_bytes": MAX_DIAGNOSTIC_TOTAL_BYTES,
                "open_frontiers": tuple(view.open_frontier_ids),
            },
            claimed_tree_id=roots.tree_id,
        )
        bridge_class = _import_symbol(
            "ipfs_accelerate_py.agent_supervisor.analysis.doctor_contract_adapters",
            "DiagnosisObligationBridge",
        )
        bridge = bridge_class.from_diagnostic_snapshot(
            diag_snapshot,
            require_repository_id=roots.repository_id,
            notes=("runtime_exact_checkout",),
        )
        if bridge.snapshot_bridge is None:  # pragma: no cover - constructor invariant
            raise DeterministicDoctorRuntimeError(
                "snapshot_bridge_missing", "diagnostic bridge omitted the snapshot"
            )
        snapshot = bridge.snapshot_bridge.materialize_deterministic()
        findings = bridge.snapshot_bridge.materialize_finding_deterministics()
        return diag_snapshot, snapshot, findings, bridge.content_id

    def _diagnose_backend(
        self,
        request: DoctorOperationRequest,
        *,
        policy: DeterministicDoctorPolicy,
        policy_decision: Any,
    ) -> DeterministicDoctorRunReceipt:
        del policy_decision
        evidence = self.build_evidence()
        snapshot = evidence.snapshot
        return DeterministicDoctorRunReceipt(
            roots=snapshot.roots,
            receipt_id=content_identity(
                {
                    "runtime": self.INTERFACE,
                    "operation": DoctorOperation.INSPECT.value,
                    "request_id": request.request_id,
                    "evidence_id": evidence.evidence_id,
                }
            ),
            operation=DoctorOperation.INSPECT,
            mode=request.mode,
            disposition=DoctorRepairDisposition.SUPPORTED,
            snapshot_id=snapshot.snapshot_id,
            incident_id=request.incident_cid(),
            network_denied=True,
            secrets_inherited=False,
            reason_codes=(
                "runtime_exact_evidence",
                "optional_providers_not_required",
            ),
            invalidation_refs=snapshot.invalidation_refs,
            resource_bounds=policy.resource_bounds,
        )

    def _plan_backend(
        self,
        request: DoctorOperationRequest,
        *,
        policy: DeterministicDoctorPolicy,
        policy_decision: Any,
    ) -> DoctorOperationResult:
        """Lazily wire analytical planning stages, then abstain if inputs are open."""

        del policy
        evidence = self.build_evidence()
        receipts: dict[str, Mapping[str, Any]] = {}
        unavailable: list[str] = []
        for stage in (
            DoctorRuntimeStage.RETRIEVE,
            DoctorRuntimeStage.TACTICIAN,
            DoctorRuntimeStage.PROOF,
            DoctorRuntimeStage.SYNTHESIS_PREVIEW,
            DoctorRuntimeStage.IMPACT,
        ):
            try:
                self._factory.get(stage)
                receipts[stage.value] = {
                    "status": "wired",
                    "reason_code": "awaiting_typed_stage_inputs",
                    "remediation": _STAGE_REMEDIATIONS[stage],
                }
            except DoctorRuntimeStageUnavailable as exc:
                unavailable.append(stage.value)
                receipts[stage.value] = {
                    "status": "unavailable",
                    "reason_code": exc.reason_code,
                    "remediation": exc.remediation,
                }
        self._stage_receipts.update(receipts)
        reasons = (
            DoctorServiceCapabilityCode.STAGE_BACKEND_MISSING.value,
            "plan_inputs_deferred",
            *(f"stage_unavailable:{name}" for name in unavailable),
        )
        return DoctorOperationResult(
            request_id=request.request_id,
            operation=DoctorOperation.PLAN.value,
            mode=request.mode,
            disposition=DoctorRepairDisposition.ABSTAIN,
            incident_id=request.incident_cid(),
            read_only=True,
            policy_decision=policy_decision,
            reason_codes=reasons,
            explanation=(
                "planning stages are lazily wired; typed retrieval/proof/operator "
                "inputs are required before an admitted plan can be materialized"
            ),
            changed=False,
            status={
                "snapshot_id": evidence.snapshot.snapshot_id,
                "capability_graph": self.capability_graph(),
                "automatic_fallback": False,
            },
            stage_refs={
                name: str(value.get("reason_code", ""))
                for name, value in receipts.items()
            },
        )

    def _deferred_stage_backend(
        self, stage: DoctorRuntimeStage
    ) -> Callable[..., DoctorOperationResult]:
        def backend(
            request: DoctorOperationRequest,
            *,
            policy: DeterministicDoctorPolicy,
            policy_decision: Any,
        ) -> DoctorOperationResult:
            del policy
            try:
                self._factory.get(stage)
                # Typed inputs still open: unavailable, never successful.
                reason = "stage_unavailable_awaiting_typed_inputs"
                remediation = _STAGE_REMEDIATIONS[stage]
            except DoctorRuntimeStageUnavailable as exc:
                reason = exc.reason_code
                remediation = exc.remediation
            self._stage_receipts[stage.value] = {
                "status": "unavailable",
                "reason_code": reason,
                "remediation": remediation,
            }
            return DoctorOperationResult(
                request_id=request.request_id,
                operation=request.operation,
                mode=request.mode,
                disposition=DoctorRepairDisposition.ABSTAIN,
                incident_id=request.incident_cid(),
                read_only=request.is_read_only,
                policy_decision=policy_decision,
                reason_codes=(
                    DoctorServiceCapabilityCode.CAPABILITY_UNAVAILABLE.value,
                    reason,
                ),
                explanation=(
                    f"{stage.value} unavailable (not successful): {remediation}"
                ),
                changed=False,
                status={
                    "stage": stage.value,
                    "automatic_fallback": False,
                    "production_success": False,
                },
                stage_refs={stage.value: reason},
            )

        # Mark optional deferred adapters so production composition can reject
        # them as mandatory-backend candidates.
        setattr(backend, "doctor_deferred_backend", True)
        setattr(backend, "doctor_stage_name", stage.value)
        return backend

    def _mandatory_stage_backend(
        self, stage: DoctorRuntimeStage
    ) -> Callable[..., DoctorOperationResult]:
        """Bind a mandatory stage: wire the class, abstain only on open inputs."""

        def backend(
            request: DoctorOperationRequest,
            *,
            policy: DeterministicDoctorPolicy,
            policy_decision: Any,
        ) -> DoctorOperationResult:
            del policy
            try:
                self._factory.get(stage)
                reason = "awaiting_typed_stage_inputs"
                remediation = _STAGE_REMEDIATIONS[stage]
                status = "wired"
            except DoctorRuntimeStageUnavailable as exc:
                # Dependency gap is typed unavailability, never empty success.
                reason = exc.reason_code
                remediation = exc.remediation
                status = "unavailable"
            self._stage_receipts[stage.value] = {
                "status": status,
                "reason_code": reason,
                "remediation": remediation,
            }
            return DoctorOperationResult(
                request_id=request.request_id,
                operation=request.operation,
                mode=request.mode,
                disposition=DoctorRepairDisposition.ABSTAIN,
                incident_id=request.incident_cid(),
                read_only=request.is_read_only,
                policy_decision=policy_decision,
                reason_codes=(
                    DoctorServiceCapabilityCode.CAPABILITY_UNAVAILABLE.value
                    if status == "unavailable"
                    else DoctorServiceCapabilityCode.STAGE_BACKEND_MISSING.value,
                    reason,
                ),
                explanation=(
                    f"{stage.value} bound; {remediation}"
                    if status == "wired"
                    else f"{stage.value} unavailable: {remediation}"
                ),
                changed=False,
                status={
                    "stage": stage.value,
                    "automatic_fallback": False,
                    "production_success": False,
                    "mandatory": True,
                    "deferred": False,
                },
                stage_refs={stage.value: reason},
            )

        setattr(backend, "doctor_deferred_backend", False)
        setattr(backend, "doctor_stage_name", stage.value)
        setattr(backend, "doctor_mandatory_backend", True)
        return backend

    def _retrieve_backend(
        self,
        request: DoctorOperationRequest,
        *,
        policy: DeterministicDoctorPolicy,
        policy_decision: Any,
    ) -> DoctorOperationResult:
        return self._mandatory_stage_backend(DoctorRuntimeStage.RETRIEVE)(
            request, policy=policy, policy_decision=policy_decision
        )

    def _tactician_backend(
        self,
        request: DoctorOperationRequest,
        *,
        policy: DeterministicDoctorPolicy,
        policy_decision: Any,
    ) -> DoctorOperationResult:
        return self._mandatory_stage_backend(DoctorRuntimeStage.TACTICIAN)(
            request, policy=policy, policy_decision=policy_decision
        )

    def _proof_backend(
        self,
        request: DoctorOperationRequest,
        *,
        policy: DeterministicDoctorPolicy,
        policy_decision: Any,
    ) -> DoctorOperationResult:
        return self._mandatory_stage_backend(DoctorRuntimeStage.PROOF)(
            request, policy=policy, policy_decision=policy_decision
        )

    def _transaction_backend(
        self,
        request: DoctorOperationRequest,
        *,
        policy: DeterministicDoctorPolicy,
        policy_decision: Any,
    ) -> DoctorOperationResult:
        # Loading the class proves wiring.  Mutation still needs real adapters
        # and the service's control dependency — never claim success without them.
        del policy
        try:
            self._factory.get(DoctorRuntimeStage.TRANSACTION)
            wired = True
            reason = "awaiting_typed_stage_inputs"
            remediation = _STAGE_REMEDIATIONS[DoctorRuntimeStage.TRANSACTION]
        except DoctorRuntimeStageUnavailable as exc:
            wired = False
            reason = exc.reason_code
            remediation = exc.remediation
        if self._control_service is None and wired:
            reason = "control_service_required"
            remediation = (
                "bind a control-plane permit/effect adapter before transaction apply"
            )
            wired = False
        status = "wired" if wired else "unavailable"
        self._stage_receipts[DoctorRuntimeStage.TRANSACTION.value] = {
            "status": status,
            "reason_code": reason,
            "remediation": remediation,
        }
        return DoctorOperationResult(
            request_id=request.request_id,
            operation=request.operation,
            mode=request.mode,
            disposition=DoctorRepairDisposition.ABSTAIN,
            incident_id=request.incident_cid(),
            read_only=request.is_read_only,
            policy_decision=policy_decision,
            reason_codes=(
                DoctorServiceCapabilityCode.CAPABILITY_UNAVAILABLE.value,
                reason,
            ),
            explanation=f"transaction {status}: {remediation}",
            changed=False,
            status={
                "stage": DoctorRuntimeStage.TRANSACTION.value,
                "automatic_fallback": False,
                "production_success": False,
                "mandatory": True,
                "deferred": False,
            },
            stage_refs={DoctorRuntimeStage.TRANSACTION.value: reason},
        )


def create_deterministic_doctor_runtime(
    checkout_root: str | os.PathLike[str],
    **kwargs: Any,
) -> DeterministicDoctorRuntime:
    """Create a cold runtime bound to one explicit checkout."""

    return DeterministicDoctorRuntime(checkout_root=checkout_root, **kwargs)


__all__ = [
    "DETERMINISTIC_DOCTOR_BACKEND_FACTORY_INTERFACE",
    "DETERMINISTIC_DOCTOR_EVIDENCE_BUNDLE_SCHEMA",
    "DETERMINISTIC_DOCTOR_RUNTIME_DISCOVERY_SCHEMA",
    "DETERMINISTIC_DOCTOR_RUNTIME_INTERFACE",
    "DETERMINISTIC_DOCTOR_RUNTIME_REPORT_SCHEMA",
    "DeterministicDoctorBackendFactory",
    "DeterministicDoctorEvidenceBundle",
    "DeterministicDoctorRuntime",
    "DeterministicDoctorRuntimeError",
    "DeterministicDoctorRuntimeReport",
    "DeterministicDoctorRuntimeSafetyError",
    "DoctorRuntimeStage",
    "DoctorRuntimeStageUnavailable",
    "DoctorSourceInventoryEntry",
    "DoctorStageCapability",
    "create_deterministic_doctor_runtime",
]
