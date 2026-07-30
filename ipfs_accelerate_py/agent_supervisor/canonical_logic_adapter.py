"""Thin vocabulary adapter between the agent supervisor and canonical datasets logic.

``SupervisorCanonicalLogicAdapter@1`` replaces overlapping family, property,
translation, matrix, capability, provider, route, resource, cache, and receipt
vocabularies with lossless projections onto the datasets logic platform while
retaining supervisor-owned scheduling, isolation, routing, cache, and evidence
behavior.

Importing this module never imports ``ipfs_datasets_py``.  Datasets packages are
loaded only for an explicit conversion, registry discovery, or revision check.
Supervisor-local facades (analysis registry, multi-prover router, proof
provider, Hammer adapter) remain the public compatibility surface; this module
is the single cross-package boundary for vocabulary identity.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from .analysis.analysis_operation_registry import (
    CacheScope,
    LogicFamily,
    normalize_logic_family,
)
from .logic_provider_contract import (
    CANONICAL_LOGIC_PROVIDER_MODULE,
    SupervisorLogicProviderFacade,
    to_logic_provider_request,
    to_supervisor_provider_response,
)
from .proof.formal_verification_capabilities import (
    FormalVerificationCapabilityReport,
    FormalVerificationProviderCapability,
    ProofProviderCapability,
    ProofProviderIsolation,
    ProofProviderOperation,
)
from .proof.formal_verification_contracts import ResourceBudget
from .proof.formal_verification_provider import (
    CancellationToken,
    ProviderRequest,
    ProviderResponse,
)
from .proof.logic_translation_validation import (
    ApproximationDirection,
    LogicForm,
    TranslationClass,
    TranslationContract,
    TranslationValidationResult,
)
from .proof.multi_prover_router import (
    PortfolioPlan,
    PropertyKind,
    PropertyObligation,
    ProverLane,
    ProverRole,
    classify_property_kind,
)
from .proof.prover_matrix_registry import ProverMatrixEntry, ProverMatrixSnapshot


SUPERVISOR_CANONICAL_LOGIC_ADAPTER_INTERFACE: Final = (
    "SupervisorCanonicalLogicAdapter@1"
)
SUPERVISOR_CANONICAL_LOGIC_ADAPTER_VERSION: Final = "1.0.0"
ADAPTER_SCHEMA_VERSION: Final = (
    "ipfs_accelerate_py/agent-supervisor/canonical-logic-adapter@1"
)
VOCABULARY_PROJECTION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/canonical-vocabulary-projection@1"
)

CANONICAL_FAMILY_REGISTRY_MODULE: Final = "ipfs_datasets_py.logic.families.registry"
CANONICAL_FAMILY_MODELS_MODULE: Final = "ipfs_datasets_py.logic.families.models"
CANONICAL_SOFTWARE_PROPERTY_MODULE: Final = (
    "ipfs_datasets_py.logic.software_verification.properties"
)
CANONICAL_TRANSLATION_MODULE: Final = (
    "ipfs_datasets_py.logic.software_verification.translations"
)
CANONICAL_RECEIPT_MODULE: Final = (
    "ipfs_datasets_py.logic.software_verification.receipts"
)
CANONICAL_CACHE_MODULE: Final = "ipfs_datasets_py.logic.backends.cache_protocol"
CANONICAL_PROVIDER_MODULE: Final = CANONICAL_LOGIC_PROVIDER_MODULE
CANONICAL_VERIFICATION_API_MODULE: Final = (
    "ipfs_datasets_py.logic.verification_api"
)

# Supervisor analysis families → datasets family_id (identity when shared).
# Values that only exist on the supervisor side use the reserved
# ``supervisor.<name>`` namespace so reverse mapping is exact.
_ANALYSIS_FAMILY_TO_CANONICAL: Final[Mapping[str, str]] = MappingProxyType(
    {
        LogicFamily.TDFOL.value: "tdfol",
        LogicFamily.DCEC.value: "dcec",
        LogicFamily.FLOGIC.value: "frame_logic",
        LogicFamily.MODAL.value: "modal",
        LogicFamily.DEONTIC.value: "deontic",
        LogicFamily.FRAME.value: "frame_logic",
        LogicFamily.KNOWLEDGE_GRAPH.value: "supervisor.kg",
        LogicFamily.EVENT_CALCULUS.value: "event_calculus",
    }
)

# Canonical family_id → preferred supervisor LogicFamily value for reverse map
# when no residual supervisor identity is present.
_CANONICAL_FAMILY_TO_SUPERVISOR: Final[Mapping[str, str]] = MappingProxyType(
    {
        "tdfol": LogicFamily.TDFOL.value,
        "dcec": LogicFamily.DCEC.value,
        "frame_logic": LogicFamily.FLOGIC.value,
        "modal": LogicFamily.MODAL.value,
        "deontic": LogicFamily.DEONTIC.value,
        "event_calculus": LogicFamily.EVENT_CALCULUS.value,
        "supervisor.kg": LogicFamily.KNOWLEDGE_GRAPH.value,
        "first_order": "first_order",  # hammer/translation only; no analysis enum
        "hyperproperty": "hyperproperty",
        "temporal": "temporal",
        "authorization": "authorization",
        "transition_system": "transition_system",
        "program": "program",
        "concurrency": "concurrency",
        "cryptographic_protocol": "cryptographic_protocol",
        "datalog": "datalog",
        "propositional": "propositional",
        "higher_order": "higher_order",
        "horn_chc": "horn_chc",
        "mu_calculus": "mu_calculus",
        "refinement": "refinement",
        "separation_logic": "separation_logic",
    }
)

# Supervisor multi-prover property kinds → datasets software-verification kinds.
_PROPERTY_KIND_TO_CANONICAL: Final[Mapping[str, str]] = MappingProxyType(
    {
        PropertyKind.FINITE_CONSTRAINT.value: "satisfiability",
        PropertyKind.STATE_MACHINE.value: "reachability",
        PropertyKind.AUTHORIZATION.value: "authorization",
        PropertyKind.PROTOCOL.value: "trace_conformance",
        PropertyKind.HYPERPROPERTY.value: "hyperproperty",
        PropertyKind.RUNTIME_TRACE.value: "trace_conformance",
        PropertyKind.KERNEL_CHECK.value: "theorem",
        PropertyKind.TYPED_PLANNING.value: "invariant",
        PropertyKind.TEMPORAL_DEONTIC.value: "safety",
        PropertyKind.FIRST_ORDER_THEOREM.value: "theorem",
    }
)

_CANONICAL_PROPERTY_TO_SUPERVISOR: Final[Mapping[str, str]] = MappingProxyType(
    {
        "satisfiability": PropertyKind.FINITE_CONSTRAINT.value,
        "reachability": PropertyKind.STATE_MACHINE.value,
        "authorization": PropertyKind.AUTHORIZATION.value,
        "trace_conformance": PropertyKind.RUNTIME_TRACE.value,
        "hyperproperty": PropertyKind.HYPERPROPERTY.value,
        "theorem": PropertyKind.FIRST_ORDER_THEOREM.value,
        "invariant": PropertyKind.TYPED_PLANNING.value,
        "safety": PropertyKind.TEMPORAL_DEONTIC.value,
        "authentication": PropertyKind.AUTHORIZATION.value,
        "noninterference": PropertyKind.HYPERPROPERTY.value,
        "liveness": PropertyKind.STATE_MACHINE.value,
        "secrecy": PropertyKind.PROTOCOL.value,
        "contract": PropertyKind.FINITE_CONSTRAINT.value,
        "data_race_freedom": PropertyKind.STATE_MACHINE.value,
        "heap_safety": PropertyKind.KERNEL_CHECK.value,
        "refinement": PropertyKind.TYPED_PLANNING.value,
        "termination": PropertyKind.STATE_MACHINE.value,
        "validity": PropertyKind.FIRST_ORDER_THEOREM.value,
    }
)

# Supervisor translation class → datasets PreservationKind / TranslationKind.
_TRANSLATION_CLASS_TO_PRESERVATION: Final[Mapping[str, str]] = MappingProxyType(
    {
        TranslationClass.EXACT.value: "exact",
        TranslationClass.EQUISATISFIABLE.value: "equisatisfiable",
        TranslationClass.BOUNDED_ABSTRACTION.value: "bounded",
        TranslationClass.CONSERVATIVE_APPROXIMATION.value: "conservative",
        TranslationClass.HEURISTIC.value: "heuristic",
    }
)

_PRESERVATION_TO_TRANSLATION_CLASS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "exact": TranslationClass.EXACT.value,
        "lossless": TranslationClass.EXACT.value,
        "equisatisfiable": TranslationClass.EQUISATISFIABLE.value,
        "bounded": TranslationClass.BOUNDED_ABSTRACTION.value,
        "bounded_abstraction": TranslationClass.BOUNDED_ABSTRACTION.value,
        "conservative": TranslationClass.CONSERVATIVE_APPROXIMATION.value,
        "conservative_approximation": TranslationClass.CONSERVATIVE_APPROXIMATION.value,
        "approximate": TranslationClass.HEURISTIC.value,
        "heuristic": TranslationClass.HEURISTIC.value,
    }
)

_TRANSLATION_CLASS_TO_TAXONOMY_KIND: Final[Mapping[str, str]] = MappingProxyType(
    {
        TranslationClass.EXACT.value: "lossless",
        TranslationClass.EQUISATISFIABLE.value: "equisatisfiable",
        TranslationClass.BOUNDED_ABSTRACTION.value: "sound_over_approximation",
        TranslationClass.CONSERVATIVE_APPROXIMATION.value: "sound_over_approximation",
        TranslationClass.HEURISTIC.value: "heuristic",
    }
)

# Supervisor LogicForm → canonical form / family labels used on wire.
_LOGIC_FORM_TO_CANONICAL: Final[Mapping[str, str]] = MappingProxyType(
    {
        LogicForm.AST.value: "ast",
        LogicForm.DCEC.value: "dcec",
        LogicForm.TDFOL.value: "tdfol",
        LogicForm.FOL.value: "first_order",
        LogicForm.TPTP.value: "tptp",
        LogicForm.SMT_LIB.value: "smtlib",
        LogicForm.TLA_PLUS.value: "transition_system",
        LogicForm.PROTOCOL.value: "cryptographic_protocol",
        LogicForm.HYPERPROPERTY.value: "hyperproperty",
    }
)

_CANONICAL_FORM_TO_LOGIC_FORM: Final[Mapping[str, str]] = MappingProxyType(
    {
        "ast": LogicForm.AST.value,
        "dcec": LogicForm.DCEC.value,
        "tdfol": LogicForm.TDFOL.value,
        "first_order": LogicForm.FOL.value,
        "fol": LogicForm.FOL.value,
        "tptp": LogicForm.TPTP.value,
        "smtlib": LogicForm.SMT_LIB.value,
        "smt-lib": LogicForm.SMT_LIB.value,
        "smtlib2": LogicForm.SMT_LIB.value,
        "transition_system": LogicForm.TLA_PLUS.value,
        "tla+": LogicForm.TLA_PLUS.value,
        "tla_plus": LogicForm.TLA_PLUS.value,
        "cryptographic_protocol": LogicForm.PROTOCOL.value,
        "protocol": LogicForm.PROTOCOL.value,
        "hyperproperty": LogicForm.HYPERPROPERTY.value,
    }
)

_CACHE_SCOPE_TO_CANONICAL: Final[Mapping[str, str]] = MappingProxyType(
    {
        CacheScope.TREE.value: "tree",
        CacheScope.OBJECTIVE.value: "policy",
        CacheScope.REQUEST.value: "request",
        CacheScope.NONE.value: "none",
    }
)

_CANONICAL_CACHE_SCOPE_TO_SUPERVISOR: Final[Mapping[str, str]] = MappingProxyType(
    {
        "tree": CacheScope.TREE.value,
        "exact_tree": CacheScope.TREE.value,
        "policy": CacheScope.OBJECTIVE.value,
        "objective_revision": CacheScope.OBJECTIVE.value,
        "request": CacheScope.REQUEST.value,
        "none": CacheScope.NONE.value,
    }
)

_ISOLATION_TO_RUNTIME: Final[Mapping[str, str]] = MappingProxyType(
    {
        ProofProviderIsolation.IN_PROCESS.value: "in_process",
        ProofProviderIsolation.SUBPROCESS.value: "native_process",
    }
)

_RUNTIME_TO_ISOLATION: Final[Mapping[str, str]] = MappingProxyType(
    {
        "in_process": ProofProviderIsolation.IN_PROCESS.value,
        "native_process": ProofProviderIsolation.SUBPROCESS.value,
        "jvm": ProofProviderIsolation.SUBPROCESS.value,
        "ocaml": ProofProviderIsolation.SUBPROCESS.value,
        "wasm": ProofProviderIsolation.SUBPROCESS.value,
        "remote_service": ProofProviderIsolation.SUBPROCESS.value,
        "declaration_only": ProofProviderIsolation.IN_PROCESS.value,
    }
)

_RESOURCE_BUDGET_FIELDS: Final[tuple[str, ...]] = (
    "wall_time_ms",
    "cpu_time_ms",
    "memory_bytes",
    "disk_bytes",
    "max_processes",
    "max_premises",
    "max_output_bytes",
    "model_token_limit",
    "provider_quota",
    "network_allowed",
)

_REQUIRED_CANONICAL_MODULES: Final[tuple[str, ...]] = (
    "ipfs_datasets_py.logic",
    "ipfs_datasets_py.logic.ir_core",
    "ipfs_datasets_py.logic.backends",
    "ipfs_datasets_py.logic.families",
    "ipfs_datasets_py.logic.software_verification",
    "ipfs_datasets_py.logic.backends.provider",
    "ipfs_datasets_py.logic.verification_api",
)

_import_lock: Final = threading.Lock()
_import_cache: dict[str, Any] = {}


class CanonicalLogicAdapterError(ValueError):
    """Raised when a vocabulary projection cannot be performed losslessly."""


@dataclass(frozen=True)
class VocabularyProjection:
    """Lossless projection of one supervisor vocabulary token onto canonical space.

    ``residual`` always retains the exact supervisor token so reverse mapping is
    identity even when multiple supervisor tokens collapse onto one canonical id
    (for example ``flogic`` and ``frame`` both project to ``frame_logic``).
    """

    domain: str
    supervisor_id: str
    canonical_id: str
    bidirectional: bool = True
    residual: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = VOCABULARY_PROJECTION_SCHEMA

    def __post_init__(self) -> None:
        if not self.domain or not isinstance(self.domain, str):
            raise CanonicalLogicAdapterError("projection domain must be a non-empty string")
        if not self.supervisor_id or not isinstance(self.supervisor_id, str):
            raise CanonicalLogicAdapterError(
                "projection supervisor_id must be a non-empty string"
            )
        if not self.canonical_id or not isinstance(self.canonical_id, str):
            raise CanonicalLogicAdapterError(
                "projection canonical_id must be a non-empty string"
            )
        if self.schema_version != VOCABULARY_PROJECTION_SCHEMA:
            raise CanonicalLogicAdapterError("unsupported vocabulary projection schema")
        if not isinstance(self.bidirectional, bool):
            raise CanonicalLogicAdapterError("bidirectional must be a boolean")
        residual = dict(self.residual or {})
        residual.setdefault("supervisor_id", self.supervisor_id)
        residual.setdefault("domain", self.domain)
        object.__setattr__(self, "residual", MappingProxyType(residual))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "domain": self.domain,
            "supervisor_id": self.supervisor_id,
            "canonical_id": self.canonical_id,
            "bidirectional": self.bidirectional,
            "residual": dict(self.residual),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> VocabularyProjection:
        if not isinstance(value, Mapping):
            raise CanonicalLogicAdapterError("vocabulary projection must be an object")
        schema = value.get("schema_version", VOCABULARY_PROJECTION_SCHEMA)
        if schema != VOCABULARY_PROJECTION_SCHEMA:
            raise CanonicalLogicAdapterError(
                f"unsupported vocabulary projection schema: {schema!r}"
            )
        return cls(
            domain=str(value.get("domain") or ""),
            supervisor_id=str(value.get("supervisor_id") or ""),
            canonical_id=str(value.get("canonical_id") or ""),
            bidirectional=bool(value.get("bidirectional", True)),
            residual=dict(value.get("residual") or {}),
            schema_version=str(schema),
        )


@dataclass(frozen=True)
class CrossRepoRevisionReport:
    """Result of a fail-closed cross-repository revision check."""

    aligned: bool
    interface: str = SUPERVISOR_CANONICAL_LOGIC_ADAPTER_INTERFACE
    parent_commit: str = ""
    datasets_gitlink: str = ""
    datasets_embedded_head: str = ""
    required_modules: Mapping[str, bool] = field(default_factory=dict)
    diagnostics: tuple[str, ...] = ()
    schema_version: str = ADAPTER_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "interface": self.interface,
            "aligned": self.aligned,
            "parent_commit": self.parent_commit,
            "datasets_gitlink": self.datasets_gitlink,
            "datasets_embedded_head": self.datasets_embedded_head,
            "required_modules": dict(self.required_modules),
            "diagnostics": list(self.diagnostics),
        }


def _token(value: Any) -> str:
    if isinstance(value, Enum):
        return str(value.value)
    raw = str(getattr(value, "value", value)).strip()
    if not raw:
        raise CanonicalLogicAdapterError("vocabulary token must be a non-empty string")
    return raw


def _normalized_token(value: Any) -> str:
    return (
        _token(value)
        .casefold()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("+", "_plus")
    )


def _lazy_import(module_name: str) -> Any:
    """Import a datasets module only after an explicit boundary call."""

    cached = _import_cache.get(module_name)
    if cached is not None:
        return cached
    with _import_lock:
        cached = _import_cache.get(module_name)
        if cached is not None:
            return cached
        module = importlib.import_module(module_name)
        _import_cache[module_name] = module
        return module


def _clear_import_cache_for_tests() -> None:
    """Test helper: drop cached datasets imports without unloading modules."""

    with _import_lock:
        _import_cache.clear()


def _unique_enum_values(enum_type: type[Enum]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for member in enum_type:
        value = str(member.value)
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return tuple(ordered)


def _git(repository: Path, *arguments: str) -> tuple[int, str, str]:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode, completed.stdout.strip(), completed.stderr.strip()


def _repo_root_from(start: Path | None = None) -> Path:
    current = (start or Path(__file__).resolve()).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "ipfs_accelerate_py").is_dir() and (
            candidate / "ipfs_datasets_py"
        ).exists():
            return candidate
    return Path(__file__).resolve().parents[2]


class SupervisorCanonicalLogicAdapter:
    """Lossless thin adapter for supervisor ↔ canonical logic vocabularies.

    The adapter is pure data for vocabulary maps.  Optional datasets types are
    constructed only when callers request live objects, and only after a lazy
    import.
    """

    interface: Final = SUPERVISOR_CANONICAL_LOGIC_ADAPTER_INTERFACE
    version: Final = SUPERVISOR_CANONICAL_LOGIC_ADAPTER_VERSION
    schema_version: Final = ADAPTER_SCHEMA_VERSION

    def __init__(
        self,
        *,
        datasets_root: Path | str | None = None,
        repo_root: Path | str | None = None,
        module_importer: Callable[[str], Any] | None = None,
    ) -> None:
        self._repo_root = Path(repo_root) if repo_root is not None else _repo_root_from()
        if datasets_root is not None:
            self._datasets_root = Path(datasets_root)
        else:
            self._datasets_root = self._repo_root / "ipfs_datasets_py"
        self._import = module_importer or _lazy_import

    # ------------------------------------------------------------------
    # Analysis families
    # ------------------------------------------------------------------

    def project_analysis_family(self, value: Any) -> VocabularyProjection:
        family = normalize_logic_family(value)
        supervisor_id = family.value
        canonical_id = _ANALYSIS_FAMILY_TO_CANONICAL.get(supervisor_id)
        if canonical_id is None:
            raise CanonicalLogicAdapterError(
                f"unsupported analysis logic family: {supervisor_id}"
            )
        return VocabularyProjection(
            domain="analysis_family",
            supervisor_id=supervisor_id,
            canonical_id=canonical_id,
            residual={
                "supervisor_enum": "LogicFamily",
                "supervisor_member": family.name,
            },
        )

    def restore_analysis_family(
        self, projection: VocabularyProjection | Mapping[str, Any] | str
    ) -> LogicFamily:
        if isinstance(projection, str):
            residual_id = ""
            canonical_id = projection
        elif isinstance(projection, VocabularyProjection):
            residual_id = str(projection.residual.get("supervisor_id") or "")
            canonical_id = projection.canonical_id
            if projection.domain not in ("analysis_family", ""):
                raise CanonicalLogicAdapterError(
                    f"projection domain {projection.domain!r} is not analysis_family"
                )
        else:
            proj = VocabularyProjection.from_dict(projection)
            residual_id = str(proj.residual.get("supervisor_id") or "")
            canonical_id = proj.canonical_id
        token = residual_id or _CANONICAL_FAMILY_TO_SUPERVISOR.get(
            canonical_id, canonical_id
        )
        # Only restore to supervisor LogicFamily members; namespaced extensions
        # that are not analysis families fail closed.
        try:
            return normalize_logic_family(token)
        except Exception as error:
            raise CanonicalLogicAdapterError(
                f"cannot restore analysis family from {canonical_id!r}"
            ) from error

    def map_analysis_family_to_canonical(self, value: Any) -> str:
        return self.project_analysis_family(value).canonical_id

    # ------------------------------------------------------------------
    # Property kinds
    # ------------------------------------------------------------------

    def project_property_kind(self, value: Any) -> VocabularyProjection:
        try:
            if isinstance(value, PropertyKind):
                kind = value
            else:
                kind = classify_property_kind(str(getattr(value, "value", value)))
        except Exception as error:
            raise CanonicalLogicAdapterError(
                f"unsupported property kind: {value!r}"
            ) from error
        supervisor_id = kind.value
        canonical_id = _PROPERTY_KIND_TO_CANONICAL.get(supervisor_id)
        if canonical_id is None:
            raise CanonicalLogicAdapterError(
                f"unsupported property kind: {supervisor_id}"
            )
        return VocabularyProjection(
            domain="property_kind",
            supervisor_id=supervisor_id,
            canonical_id=canonical_id,
            residual={
                "supervisor_enum": "PropertyKind",
                "supervisor_member": kind.name,
            },
        )

    def restore_property_kind(
        self, projection: VocabularyProjection | Mapping[str, Any] | str
    ) -> PropertyKind:
        if isinstance(projection, str):
            residual_id = ""
            canonical_id = projection
        elif isinstance(projection, VocabularyProjection):
            residual_id = str(projection.residual.get("supervisor_id") or "")
            canonical_id = projection.canonical_id
        else:
            proj = VocabularyProjection.from_dict(projection)
            residual_id = str(proj.residual.get("supervisor_id") or "")
            canonical_id = proj.canonical_id
        token = residual_id or _CANONICAL_PROPERTY_TO_SUPERVISOR.get(
            canonical_id, canonical_id
        )
        return classify_property_kind(token)

    def map_property_kind_to_canonical(self, value: Any) -> str:
        return self.project_property_kind(value).canonical_id

    # ------------------------------------------------------------------
    # Translation forms and classes
    # ------------------------------------------------------------------

    def project_logic_form(self, value: Any) -> VocabularyProjection:
        try:
            form = value if isinstance(value, LogicForm) else LogicForm(_token(value))
        except (TypeError, ValueError) as error:
            raise CanonicalLogicAdapterError(
                f"unsupported logic form: {_token(value)}"
            ) from error
        supervisor_id = form.value
        canonical_id = _LOGIC_FORM_TO_CANONICAL.get(supervisor_id)
        if canonical_id is None:
            raise CanonicalLogicAdapterError(f"unsupported logic form: {supervisor_id}")
        return VocabularyProjection(
            domain="logic_form",
            supervisor_id=supervisor_id,
            canonical_id=canonical_id,
            residual={"supervisor_enum": "LogicForm", "supervisor_member": form.name},
        )

    def restore_logic_form(
        self, projection: VocabularyProjection | Mapping[str, Any] | str
    ) -> LogicForm:
        if isinstance(projection, str):
            residual_id = ""
            canonical_id = projection
        elif isinstance(projection, VocabularyProjection):
            residual_id = str(projection.residual.get("supervisor_id") or "")
            canonical_id = projection.canonical_id
        else:
            proj = VocabularyProjection.from_dict(projection)
            residual_id = str(proj.residual.get("supervisor_id") or "")
            canonical_id = proj.canonical_id
        token = residual_id or _CANONICAL_FORM_TO_LOGIC_FORM.get(
            canonical_id, canonical_id
        )
        try:
            return LogicForm(token)
        except ValueError as error:
            raise CanonicalLogicAdapterError(
                f"cannot restore logic form from {canonical_id!r}"
            ) from error

    def project_translation_class(self, value: Any) -> VocabularyProjection:
        try:
            translation = (
                value
                if isinstance(value, TranslationClass)
                else TranslationClass(_token(value))
            )
        except (TypeError, ValueError) as error:
            raise CanonicalLogicAdapterError(
                f"unsupported translation class: {_token(value)}"
            ) from error
        supervisor_id = translation.value
        preservation = _TRANSLATION_CLASS_TO_PRESERVATION.get(supervisor_id)
        if preservation is None:
            raise CanonicalLogicAdapterError(
                f"unsupported translation class: {supervisor_id}"
            )
        taxonomy = _TRANSLATION_CLASS_TO_TAXONOMY_KIND[supervisor_id]
        return VocabularyProjection(
            domain="translation_class",
            supervisor_id=supervisor_id,
            canonical_id=preservation,
            residual={
                "supervisor_enum": "TranslationClass",
                "supervisor_member": translation.name,
                "taxonomy_translation_kind": taxonomy,
            },
        )

    def restore_translation_class(
        self, projection: VocabularyProjection | Mapping[str, Any] | str
    ) -> TranslationClass:
        if isinstance(projection, str):
            residual_id = ""
            canonical_id = projection
        elif isinstance(projection, VocabularyProjection):
            residual_id = str(projection.residual.get("supervisor_id") or "")
            canonical_id = projection.canonical_id
        else:
            proj = VocabularyProjection.from_dict(projection)
            residual_id = str(proj.residual.get("supervisor_id") or "")
            canonical_id = proj.canonical_id
        token = residual_id or _PRESERVATION_TO_TRANSLATION_CLASS.get(
            canonical_id, canonical_id
        )
        try:
            return TranslationClass(token)
        except ValueError as error:
            raise CanonicalLogicAdapterError(
                f"cannot restore translation class from {canonical_id!r}"
            ) from error

    def project_translation_contract(
        self, contract: TranslationContract
    ) -> dict[str, Any]:
        if not isinstance(contract, TranslationContract):
            raise TypeError("contract must be a TranslationContract")
        source = self.project_logic_form(contract.source_form)
        target = self.project_logic_form(contract.target_form)
        translation = self.project_translation_class(contract.translation_class)
        direction = (
            contract.approximation_direction.value
            if isinstance(contract.approximation_direction, ApproximationDirection)
            else str(contract.approximation_direction)
        )
        return {
            "schema_version": ADAPTER_SCHEMA_VERSION,
            "interface": self.interface,
            "contract_id": contract.contract_id,
            "source_form": source.to_dict(),
            "target_form": target.to_dict(),
            "translation_class": translation.to_dict(),
            "taxonomy_translation_kind": translation.residual.get(
                "taxonomy_translation_kind"
            ),
            "approximation_direction": direction,
            "translator_id": contract.translator_id,
            "translator_version": contract.translator_version,
            "translator_identity": contract.translator_identity,
            "source_identity": contract.source_identity,
            "semantic_profile_id": contract.semantic_profile_id,
            "semantic_profile_version": contract.semantic_profile_version,
            "fixture_set_id": contract.fixture_set_id,
            "permitted_assurance": getattr(
                contract.permitted_assurance, "value", contract.permitted_assurance
            ),
            "assumptions": list(contract.assumptions),
            "required_bounds": list(contract.required_bounds),
            "abstracted_dimensions": [
                getattr(item, "value", item) for item in contract.abstracted_dimensions
            ],
            "metadata": dict(contract.metadata),
            "content_id": contract.content_id,
        }

    def restore_translation_contract_forms(
        self, payload: Mapping[str, Any]
    ) -> tuple[LogicForm, LogicForm, TranslationClass]:
        if not isinstance(payload, Mapping):
            raise CanonicalLogicAdapterError(
                "translation contract projection must be an object"
            )
        source = self.restore_logic_form(payload.get("source_form") or {})
        target = self.restore_logic_form(payload.get("target_form") or {})
        translation = self.restore_translation_class(
            payload.get("translation_class") or {}
        )
        return source, target, translation

    # ------------------------------------------------------------------
    # Matrix entries
    # ------------------------------------------------------------------

    def project_matrix_entry(self, entry: ProverMatrixEntry) -> dict[str, Any]:
        if not isinstance(entry, ProverMatrixEntry):
            raise TypeError("entry must be a ProverMatrixEntry")
        payload = entry.to_dict()
        return {
            "schema_version": ADAPTER_SCHEMA_VERSION,
            "interface": self.interface,
            "domain": "matrix_entry",
            "prover_id": entry.prover_id,
            "supervisor_entry": payload,
            "canonical_provider_id": entry.prover_id,
            "state": getattr(entry.state, "value", entry.state)
            if hasattr(entry, "state")
            else payload.get("state"),
            "authority": payload.get("authority")
            or payload.get("maximum_authoritative_for")
            or (),
            "residual": {
                "supervisor_schema": payload.get("schema")
                or payload.get("schema_version"),
            },
        }

    def restore_matrix_entry_payload(
        self, projection: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        if not isinstance(projection, Mapping):
            raise CanonicalLogicAdapterError("matrix entry projection must be an object")
        supervisor_entry = projection.get("supervisor_entry")
        if not isinstance(supervisor_entry, Mapping):
            raise CanonicalLogicAdapterError(
                "matrix entry projection is missing supervisor_entry"
            )
        return dict(supervisor_entry)

    def project_matrix_snapshot(
        self, snapshot: ProverMatrixSnapshot
    ) -> dict[str, Any]:
        if not isinstance(snapshot, ProverMatrixSnapshot):
            raise TypeError("snapshot must be a ProverMatrixSnapshot")
        return {
            "schema_version": ADAPTER_SCHEMA_VERSION,
            "interface": self.interface,
            "domain": "matrix_snapshot",
            "entries": [self.project_matrix_entry(entry) for entry in snapshot.entries],
            "supervisor_snapshot": snapshot.to_dict(),
        }

    # ------------------------------------------------------------------
    # Capability probes
    # ------------------------------------------------------------------

    def project_provider_capability(
        self, capability: ProofProviderCapability | FormalVerificationProviderCapability
    ) -> dict[str, Any]:
        if isinstance(capability, ProofProviderCapability):
            payload = capability.to_dict()
            operations = [
                getattr(item, "value", item) for item in capability.operations
            ]
            isolation = [
                getattr(item, "value", item) for item in capability.isolation
            ]
            provider_id = capability.provider_id
            provider_version = capability.provider_version
        elif isinstance(capability, FormalVerificationProviderCapability):
            payload = capability.to_dict()
            operations = list(payload.get("operations") or ())
            isolation = list(payload.get("isolation") or ())
            provider_id = str(
                payload.get("provider_id") or getattr(capability, "provider_id", "")
            )
            provider_version = str(
                payload.get("provider_version")
                or getattr(capability, "provider_version", "")
                or "unknown"
            )
        else:
            raise TypeError(
                "capability must be a ProofProviderCapability or "
                "FormalVerificationProviderCapability"
            )
        runtimes = [
            _ISOLATION_TO_RUNTIME.get(str(mode), "native_process") for mode in isolation
        ]
        return {
            "schema_version": ADAPTER_SCHEMA_VERSION,
            "interface": self.interface,
            "domain": "capability_probe",
            "provider_id": provider_id,
            "provider_version": provider_version,
            "operations": operations,
            "isolation": isolation,
            "runtimes": runtimes,
            "network_access_required": bool(
                payload.get("network_access_required", False)
            ),
            "resource_limits_supported": bool(
                payload.get("resource_limits_supported", False)
            ),
            "proof_attempted": False,
            "proof_success": False,
            "supervisor_capability": payload,
        }

    def restore_provider_capability(
        self, projection: Mapping[str, Any]
    ) -> ProofProviderCapability:
        if not isinstance(projection, Mapping):
            raise CanonicalLogicAdapterError(
                "capability projection must be an object"
            )
        supervisor = projection.get("supervisor_capability")
        if isinstance(supervisor, Mapping) and supervisor.get("provider_id"):
            # Prefer exact supervisor payload when present.
            try:
                return ProofProviderCapability(
                    provider_id=str(supervisor.get("provider_id") or ""),
                    provider_version=str(supervisor.get("provider_version") or "1"),
                    protocol_versions=tuple(
                        supervisor.get("protocol_versions") or (1,)
                    ),
                    operations=tuple(
                        supervisor.get("operations")
                        or (ProofProviderOperation.CAPABILITY,)
                    ),
                    isolation=tuple(
                        supervisor.get("isolation")
                        or (ProofProviderIsolation.IN_PROCESS,)
                    ),
                    network_access_required=bool(
                        supervisor.get("network_access_required", False)
                    ),
                    resource_limits_supported=bool(
                        supervisor.get("resource_limits_supported", False)
                    ),
                    metadata=dict(supervisor.get("metadata") or {}),
                )
            except (TypeError, ValueError):
                pass
        isolation_modes: list[str] = []
        for runtime in projection.get("runtimes") or projection.get("isolation") or ():
            token = str(runtime)
            isolation_modes.append(
                _RUNTIME_TO_ISOLATION.get(token, token)
                if token in _RUNTIME_TO_ISOLATION
                or token
                in {
                    ProofProviderIsolation.IN_PROCESS.value,
                    ProofProviderIsolation.SUBPROCESS.value,
                }
                else ProofProviderIsolation.SUBPROCESS.value
            )
        if not isolation_modes:
            isolation_modes = [ProofProviderIsolation.IN_PROCESS.value]
        operations = tuple(
            projection.get("operations") or (ProofProviderOperation.CAPABILITY.value,)
        )
        if ProofProviderOperation.CAPABILITY.value not in {
            str(getattr(item, "value", item)) for item in operations
        }:
            operations = (ProofProviderOperation.CAPABILITY.value, *operations)
        return ProofProviderCapability(
            provider_id=str(projection.get("provider_id") or ""),
            provider_version=str(projection.get("provider_version") or "1"),
            operations=operations,
            isolation=tuple(isolation_modes),
            network_access_required=bool(
                projection.get("network_access_required", False)
            ),
            resource_limits_supported=bool(
                projection.get("resource_limits_supported", False)
            ),
        )

    def project_capability_report(
        self, report: FormalVerificationCapabilityReport
    ) -> dict[str, Any]:
        if not isinstance(report, FormalVerificationCapabilityReport):
            raise TypeError("report must be a FormalVerificationCapabilityReport")
        providers = []
        for provider in report.providers:
            providers.append(self.project_provider_capability(provider))
        return {
            "schema_version": ADAPTER_SCHEMA_VERSION,
            "interface": self.interface,
            "domain": "capability_report",
            "generated_at": report.generated_at,
            "duration_seconds": report.duration_seconds,
            "probe_count": report.probe_count,
            "bounded": report.bounded,
            "proof_attempted": False,
            "proof_success": False,
            "providers": providers,
            "supervisor_report": report.to_dict(),
        }

    # ------------------------------------------------------------------
    # Providers and requests
    # ------------------------------------------------------------------

    def project_provider_request(
        self,
        request: ProviderRequest,
        *,
        cancellation: CancellationToken | None = None,
    ) -> Any:
        """Convert a supervisor request to the canonical datasets request type."""

        if not isinstance(request, ProviderRequest):
            raise TypeError("request must be a supervisor ProviderRequest")
        contract = self._import(CANONICAL_PROVIDER_MODULE)
        return to_logic_provider_request(
            request,
            cancellation=cancellation,
            contract_module=contract,
        )

    def restore_provider_response(
        self,
        request: ProviderRequest,
        response: Any,
    ) -> ProviderResponse:
        if not isinstance(request, ProviderRequest):
            raise TypeError("request must be a supervisor ProviderRequest")
        contract = self._import(CANONICAL_PROVIDER_MODULE)
        return to_supervisor_provider_response(
            request,
            response,
            contract_module=contract,
        )

    def project_resource_budget(self, budget: ResourceBudget | Mapping[str, Any]) -> dict[str, Any]:
        if isinstance(budget, ResourceBudget):
            payload = budget.to_dict()
        elif isinstance(budget, Mapping):
            payload = dict(budget)
        else:
            raise TypeError("budget must be a ResourceBudget or mapping")
        result = {
            field_name: payload.get(field_name, 0 if field_name != "network_allowed" else False)
            for field_name in _RESOURCE_BUDGET_FIELDS
        }
        result["schema_version"] = (
            "ipfs_datasets_py/logic-provider-resource-budget@1"
        )
        result["supervisor_schema"] = payload.get("schema") or payload.get(
            "schema_version"
        )
        result["interface"] = self.interface
        return result

    def restore_resource_budget(
        self, projection: Mapping[str, Any]
    ) -> ResourceBudget:
        if not isinstance(projection, Mapping):
            raise CanonicalLogicAdapterError("resource projection must be an object")
        kwargs = {
            field_name: projection.get(field_name, 0 if field_name != "network_allowed" else False)
            for field_name in _RESOURCE_BUDGET_FIELDS
        }
        return ResourceBudget(**kwargs)

    def make_logic_provider_facade(
        self,
        *,
        provider_id: str,
        provider_version: str,
        provider: Any | None = None,
        loader: Callable[[], Any] | None = None,
        target: str | None = None,
    ) -> SupervisorLogicProviderFacade:
        """Build a lazy supervisor facade over a canonical datasets provider."""

        if target is not None:
            return SupervisorLogicProviderFacade.from_reference(
                target,
                provider_id=provider_id,
                provider_version=provider_version,
            )
        return SupervisorLogicProviderFacade(
            provider_id=provider_id,
            provider_version=provider_version,
            provider=provider,
            loader=loader,
        )

    # ------------------------------------------------------------------
    # Routes / portfolio plans
    # ------------------------------------------------------------------

    def project_prover_lane(self, lane: ProverLane) -> dict[str, Any]:
        if not isinstance(lane, ProverLane):
            raise TypeError("lane must be a ProverLane")
        return {
            "schema_version": ADAPTER_SCHEMA_VERSION,
            "interface": self.interface,
            "domain": "route_lane",
            "prover_id": lane.prover_id,
            "role": lane.role.value if isinstance(lane.role, ProverRole) else str(lane.role),
            "stage": lane.stage,
            "authority_capability": lane.authority_capability,
            "translation_path_id": lane.translation_path_id,
            "requires_candidate": lane.requires_candidate,
            "supervisor_lane": lane.to_dict(),
        }

    def restore_prover_lane(self, projection: Mapping[str, Any]) -> ProverLane:
        if not isinstance(projection, Mapping):
            raise CanonicalLogicAdapterError("route projection must be an object")
        supervisor = projection.get("supervisor_lane")
        if isinstance(supervisor, Mapping):
            return ProverLane.from_dict(supervisor)
        return ProverLane(
            prover_id=str(projection.get("prover_id") or ""),
            role=str(projection.get("role") or ProverRole.CANDIDATE.value),
            stage=int(projection.get("stage") or 0),
            authority_capability=str(projection.get("authority_capability") or ""),
            translation_path_id=str(projection.get("translation_path_id") or ""),
            requires_candidate=bool(projection.get("requires_candidate", False)),
        )

    def project_portfolio_plan(self, plan: PortfolioPlan) -> dict[str, Any]:
        if not isinstance(plan, PortfolioPlan):
            raise TypeError("plan must be a PortfolioPlan")
        payload = plan.to_dict()
        lanes = []
        for lane in getattr(plan, "lanes", ()) or ():
            if isinstance(lane, ProverLane):
                lanes.append(self.project_prover_lane(lane))
            elif isinstance(lane, Mapping):
                lanes.append(self.project_prover_lane(ProverLane.from_dict(lane)))
        return {
            "schema_version": ADAPTER_SCHEMA_VERSION,
            "interface": self.interface,
            "domain": "route_plan",
            "plan_id": getattr(plan, "plan_id", payload.get("plan_id")),
            "lanes": lanes,
            "prover_ids": list(getattr(plan, "prover_ids", ()) or payload.get("prover_ids") or ()),
            "supervisor_plan": payload,
        }

    def restore_portfolio_plan_payload(
        self, projection: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        if not isinstance(projection, Mapping):
            raise CanonicalLogicAdapterError("route plan projection must be an object")
        supervisor = projection.get("supervisor_plan")
        if not isinstance(supervisor, Mapping):
            raise CanonicalLogicAdapterError(
                "route plan projection is missing supervisor_plan"
            )
        return dict(supervisor)

    def project_property_obligation(
        self, obligation: PropertyObligation
    ) -> dict[str, Any]:
        if not isinstance(obligation, PropertyObligation):
            raise TypeError("obligation must be a PropertyObligation")
        kind = self.project_property_kind(obligation.property_kind)
        return {
            "schema_version": ADAPTER_SCHEMA_VERSION,
            "interface": self.interface,
            "domain": "property_obligation",
            "obligation_id": obligation.obligation_id,
            "property_kind": kind.to_dict(),
            "statement": obligation.statement,
            "premise_ids": list(obligation.premise_ids),
            "required_assurance": getattr(
                obligation.required_assurance,
                "value",
                obligation.required_assurance,
            ),
            "metadata": dict(obligation.metadata),
            "content_id": obligation.content_id,
            "supervisor_obligation": obligation.to_dict(),
        }

    # ------------------------------------------------------------------
    # Caches
    # ------------------------------------------------------------------

    def project_cache_scope(self, value: Any) -> VocabularyProjection:
        try:
            scope = (
                value if isinstance(value, CacheScope) else CacheScope(_token(value))
            )
        except (TypeError, ValueError) as error:
            raise CanonicalLogicAdapterError(
                f"unsupported cache scope: {_token(value)}"
            ) from error
        supervisor_id = scope.value
        canonical_id = _CACHE_SCOPE_TO_CANONICAL.get(supervisor_id)
        if canonical_id is None:
            raise CanonicalLogicAdapterError(f"unsupported cache scope: {supervisor_id}")
        return VocabularyProjection(
            domain="cache_scope",
            supervisor_id=supervisor_id,
            canonical_id=canonical_id,
            residual={"supervisor_enum": "CacheScope", "supervisor_member": scope.name},
        )

    def restore_cache_scope(
        self, projection: VocabularyProjection | Mapping[str, Any] | str
    ) -> CacheScope:
        if isinstance(projection, str):
            residual_id = ""
            canonical_id = projection
        elif isinstance(projection, VocabularyProjection):
            residual_id = str(projection.residual.get("supervisor_id") or "")
            canonical_id = projection.canonical_id
        else:
            proj = VocabularyProjection.from_dict(projection)
            residual_id = str(proj.residual.get("supervisor_id") or "")
            canonical_id = proj.canonical_id
        token = residual_id or _CANONICAL_CACHE_SCOPE_TO_SUPERVISOR.get(
            canonical_id, canonical_id
        )
        try:
            return CacheScope(token)
        except ValueError as error:
            raise CanonicalLogicAdapterError(
                f"cannot restore cache scope from {canonical_id!r}"
            ) from error

    def project_proof_cache_key(self, key: Any) -> dict[str, Any]:
        """Project a supervisor proof-cache key onto the verification-cache shape."""

        if hasattr(key, "to_dict"):
            payload = key.to_dict()
        elif isinstance(key, Mapping):
            payload = dict(key)
        else:
            raise TypeError("cache key must provide to_dict() or be a mapping")

        def _digest(field_name: str, fallback: str = "") -> str:
            value = payload.get(field_name, fallback)
            if value is None or value == "":
                return fallback or f"empty:{field_name}"
            if isinstance(value, str) and value.startswith(
                ("sha256:", "proof-cache-key:", "cid:", "baguqeera")
            ):
                return value
            encoded = json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                default=str,
            )
            import hashlib

            return f"sha256:{hashlib.sha256(encoded.encode('utf-8')).hexdigest()}"

        return {
            "schema_version": "verification-cache-key/v1",
            "interface": self.interface,
            "domain": "cache_key",
            "ir_digest": _digest("obligation"),
            "property_digest": _digest("policy"),
            "assumptions_digest": _digest("premises"),
            "translation_digest": _digest("translator"),
            "backend_id": str(payload.get("solver") or payload.get("kernel") or "unknown"),
            "backend_binary_digest": _digest("toolchain"),
            "backend_version": str(payload.get("kernel") or "unknown"),
            "backend_config_digest": _digest("theorem_registry"),
            "resources_digest": _digest("resource_budget"),
            "tree_digest": _digest("candidate_tree"),
            "policy_digest": _digest("policy"),
            "supervisor_cache_key": payload,
            "supervisor_key_id": getattr(key, "key_id", None)
            or payload.get("key_id")
            or "",
        }

    def restore_proof_cache_key_payload(
        self, projection: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        if not isinstance(projection, Mapping):
            raise CanonicalLogicAdapterError("cache key projection must be an object")
        supervisor = projection.get("supervisor_cache_key")
        if not isinstance(supervisor, Mapping):
            raise CanonicalLogicAdapterError(
                "cache key projection is missing supervisor_cache_key"
            )
        return dict(supervisor)

    # ------------------------------------------------------------------
    # Receipts
    # ------------------------------------------------------------------

    def project_translation_validation_receipt(
        self, result: TranslationValidationResult | Mapping[str, Any]
    ) -> dict[str, Any]:
        if isinstance(result, TranslationValidationResult):
            payload = result.to_dict()
            content_id = result.content_id
        elif isinstance(result, Mapping):
            payload = dict(result)
            content_id = str(payload.get("content_id") or "")
        else:
            raise TypeError(
                "result must be a TranslationValidationResult or mapping"
            )
        issues = [
            dict(issue) if isinstance(issue, Mapping) else {"detail": str(issue)}
            for issue in payload.get("issues") or ()
        ]
        return {
            "schema_version": ADAPTER_SCHEMA_VERSION,
            "interface": self.interface,
            "domain": "translation_receipt",
            "valid": bool(payload.get("valid", payload.get("ok", False))),
            "issues": issues,
            "contract_identity": payload.get("contract_identity")
            or payload.get("contract_id")
            or "",
            "source_identity": payload.get("source_identity") or "",
            "target_identity": payload.get("target_identity") or "",
            "content_id": content_id,
            "supervisor_receipt": payload,
            # Never upgrade authority at the adapter boundary.
            "authority": "none",
            "proof_attempted": False,
            "proof_success": False,
        }

    def restore_translation_validation_payload(
        self, projection: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        if not isinstance(projection, Mapping):
            raise CanonicalLogicAdapterError("receipt projection must be an object")
        supervisor = projection.get("supervisor_receipt")
        if not isinstance(supervisor, Mapping):
            raise CanonicalLogicAdapterError(
                "receipt projection is missing supervisor_receipt"
            )
        return dict(supervisor)

    def load_canonical_translation_receipt_type(self) -> Any:
        """Lazy-load the datasets LogicTranslationReceipt type."""

        module = self._import(CANONICAL_RECEIPT_MODULE)
        return getattr(module, "LogicTranslationReceipt")

    # ------------------------------------------------------------------
    # Datasets discovery (lazy)
    # ------------------------------------------------------------------

    def load_canonical_family_registry(self) -> Any:
        module = self._import(CANONICAL_FAMILY_REGISTRY_MODULE)
        return getattr(module, "DEFAULT_REGISTRY", None) or getattr(
            module, "CANONICAL_REGISTRY"
        )

    def list_canonical_family_ids(self) -> tuple[str, ...]:
        registry = self.load_canonical_family_registry()
        families = getattr(registry, "families", {})
        return tuple(sorted(str(item) for item in families.keys()))

    def list_canonical_property_ids(self) -> tuple[str, ...]:
        module = self._import(CANONICAL_SOFTWARE_PROPERTY_MODULE)
        vocabulary = getattr(module, "PROPERTY_VOCABULARY", None)
        if vocabulary is not None:
            return tuple(vocabulary)
        property_kind = getattr(module, "PropertyKind")
        return tuple(sorted(item.value for item in property_kind))

    def datasets_import_is_lazy(self) -> bool:
        """Return True when datasets modules are not yet imported by this adapter."""

        return not any(
            name.startswith("ipfs_datasets_py") for name in _import_cache
        )

    # ------------------------------------------------------------------
    # Cross-repo current-revision checks
    # ------------------------------------------------------------------

    def check_cross_repo_revision(
        self,
        *,
        repo_root: Path | str | None = None,
        require_git_alignment: bool = True,
    ) -> CrossRepoRevisionReport:
        """Verify required logic modules and optional gitlink/HEAD alignment."""

        root = Path(repo_root) if repo_root is not None else self._repo_root
        diagnostics: list[str] = []
        module_status: dict[str, bool] = {}
        for module_name in _REQUIRED_CANONICAL_MODULES:
            try:
                self._import(module_name)
                module_status[module_name] = True
            except Exception as error:  # pragma: no cover - environment specific
                module_status[module_name] = False
                diagnostics.append(
                    f"module_unavailable:{module_name}:{type(error).__name__}"
                )

        parent_commit = ""
        gitlink = ""
        embedded_head = ""
        datasets_path = root / "ipfs_datasets_py"

        code, parent_commit, err = _git(root, "rev-parse", "HEAD")
        if code != 0:
            parent_commit = ""
            if require_git_alignment:
                diagnostics.append(f"parent_rev_parse_failed:{err or 'unknown'}")

        code, ls_out, err = _git(root, "ls-files", "-s", "ipfs_datasets_py")
        if code == 0 and ls_out:
            # gitlink mode 160000: <mode> <sha> <stage>\tpath
            parts = ls_out.split()
            if len(parts) >= 2 and parts[0] == "160000":
                gitlink = parts[1]
            elif datasets_path.is_dir():
                diagnostics.append("datasets_path_is_not_gitlink")
        elif require_git_alignment:
            diagnostics.append(f"gitlink_lookup_failed:{err or 'missing'}")

        if datasets_path.is_dir() and (datasets_path / ".git").exists() or (
            datasets_path.is_dir()
            and (root / ".git").exists()
        ):
            code, embedded_head, err = _git(datasets_path, "rev-parse", "HEAD")
            if code != 0:
                # Nested worktree / plain directory without independent git.
                embedded_head = ""
                if gitlink:
                    diagnostics.append(
                        f"embedded_rev_parse_failed:{err or 'unknown'}"
                    )

        aligned = all(module_status.values())
        if require_git_alignment and gitlink and embedded_head and gitlink != embedded_head:
            aligned = False
            diagnostics.append(
                f"gitlink_mismatch:gitlink={gitlink}:embedded={embedded_head}"
            )
        if diagnostics and not all(module_status.values()):
            aligned = False
        if require_git_alignment and any(
            item.startswith("parent_rev_parse_failed")
            or item.startswith("gitlink_lookup_failed")
            for item in diagnostics
        ):
            # Missing git metadata is not a hard fail when modules resolve —
            # checkout may be an exported tree.  Modules are the hard gate.
            if all(module_status.values()):
                aligned = True

        return CrossRepoRevisionReport(
            aligned=aligned,
            parent_commit=parent_commit,
            datasets_gitlink=gitlink,
            datasets_embedded_head=embedded_head,
            required_modules=MappingProxyType(module_status),
            diagnostics=tuple(diagnostics),
        )

    # ------------------------------------------------------------------
    # Aggregate inventory
    # ------------------------------------------------------------------

    def vocabulary_inventory(self) -> dict[str, Any]:
        """Return the closed supervisor vocabulary sets covered by this adapter."""

        return {
            "schema_version": self.schema_version,
            "interface": self.interface,
            "version": self.version,
            "analysis_families": list(_unique_enum_values(LogicFamily)),
            "property_kinds": list(_unique_enum_values(PropertyKind)),
            "logic_forms": list(_unique_enum_values(LogicForm)),
            "translation_classes": list(_unique_enum_values(TranslationClass)),
            "cache_scopes": list(_unique_enum_values(CacheScope)),
            "provider_operations": list(_unique_enum_values(ProofProviderOperation)),
            "domains": [
                "analysis_family",
                "property_kind",
                "logic_form",
                "translation_class",
                "matrix_entry",
                "capability_probe",
                "provider",
                "route",
                "resource",
                "cache",
                "receipt",
            ],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "interface": self.interface,
            "version": self.version,
            "vocabulary": self.vocabulary_inventory(),
            "datasets_root": str(self._datasets_root),
            "repo_root": str(self._repo_root),
        }


# Module-level singleton helpers for facade compatibility.
_default_adapter: SupervisorCanonicalLogicAdapter | None = None
_default_adapter_lock = threading.Lock()


def get_canonical_logic_adapter(
    **kwargs: Any,
) -> SupervisorCanonicalLogicAdapter:
    """Return a process-wide adapter, or a fresh one when overrides are supplied."""

    if kwargs:
        return SupervisorCanonicalLogicAdapter(**kwargs)
    global _default_adapter
    adapter = _default_adapter
    if adapter is None:
        with _default_adapter_lock:
            adapter = _default_adapter
            if adapter is None:
                adapter = SupervisorCanonicalLogicAdapter()
                _default_adapter = adapter
    return adapter


def map_analysis_family_to_canonical(value: Any) -> str:
    return get_canonical_logic_adapter().map_analysis_family_to_canonical(value)


def map_property_kind_to_canonical(value: Any) -> str:
    return get_canonical_logic_adapter().map_property_kind_to_canonical(value)


def project_resource_budget(budget: ResourceBudget | Mapping[str, Any]) -> dict[str, Any]:
    return get_canonical_logic_adapter().project_resource_budget(budget)


def project_provider_request(
    request: ProviderRequest,
    *,
    cancellation: CancellationToken | None = None,
) -> Any:
    return get_canonical_logic_adapter().project_provider_request(
        request, cancellation=cancellation
    )


def check_cross_repo_revision(**kwargs: Any) -> CrossRepoRevisionReport:
    return get_canonical_logic_adapter().check_cross_repo_revision(**kwargs)


__all__ = [
    "ADAPTER_SCHEMA_VERSION",
    "CANONICAL_CACHE_MODULE",
    "CANONICAL_FAMILY_MODELS_MODULE",
    "CANONICAL_FAMILY_REGISTRY_MODULE",
    "CANONICAL_PROVIDER_MODULE",
    "CANONICAL_RECEIPT_MODULE",
    "CANONICAL_SOFTWARE_PROPERTY_MODULE",
    "CANONICAL_TRANSLATION_MODULE",
    "CANONICAL_VERIFICATION_API_MODULE",
    "CanonicalLogicAdapterError",
    "CrossRepoRevisionReport",
    "SUPERVISOR_CANONICAL_LOGIC_ADAPTER_INTERFACE",
    "SUPERVISOR_CANONICAL_LOGIC_ADAPTER_VERSION",
    "SupervisorCanonicalLogicAdapter",
    "VOCABULARY_PROJECTION_SCHEMA",
    "VocabularyProjection",
    "check_cross_repo_revision",
    "get_canonical_logic_adapter",
    "map_analysis_family_to_canonical",
    "map_property_kind_to_canonical",
    "project_provider_request",
    "project_resource_budget",
]
