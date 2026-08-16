"""Wire Intent / Legal / Security / UI IR surfaces into the agent supervisor.

This module is the SCA-facing inventory + readiness probe for the shared-IR
stack already owned by:

* :mod:`proof.ir_registry` — ``IRFamily`` (ir_core, formalization, intent_ir,
  legal_ir, security_ir)
* :mod:`proof.ir_adapters` — provider-free normalization adapters
* :mod:`proof.intent_constraint_adapter` / ``legal_constraint_adapter`` /
  ``security_constraint_adapter`` — constraint compilation (no execution grant)
* :mod:`proof.ir_constraint_compiler` — plan admission over IR constraints
* :mod:`proof.interface_contract_codegen` — UI/interface action-contract bridge
  (stand-in until ``ipfs_datasets_py.logic.ui_ux_ir`` lands)

Authority rules (fail-closed):

* Intent describes required work; it never authorizes execution.
* Legal applicability is context; it never grants capability.
* Security decisions alone authorize; inventory does not invent grants.
* UI/interface contracts are descriptors only; they never become KERNEL_VERIFIED.
* Missing ``ui_ux_ir`` package is reported as unavailable, not forged.

LLM implement remains proposal_only under RPR.
"""

from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass, field
from typing import Any, Final, Mapping, Sequence


SCA_IR_INTEGRATION_INTERFACE: Final = "ScaIrIntegration@1"
SCA_IR_INTEGRATION_REPORT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/sca-ir-integration@1"
)

# Closed SCA analysis family names bound into multi-family / planning portfolios
DEFAULT_IR_ANALYSIS_FAMILIES: Final[tuple[str, ...]] = (
    "intent_ir",
    "legal_ir",
    "security_ir",
    "ui_ir",
)

# Registry-backed families (authoritative shared-IR enum)
DEFAULT_REGISTRY_IR_FAMILIES: Final[tuple[str, ...]] = (
    "ir_core",
    "formalization",
    "intent_ir",
    "legal_ir",
    "security_ir",
    "ui_ir",
)

# Datasets logic packages expected for full IR body sources
DEFAULT_DATASETS_IR_PACKAGES: Final[tuple[str, ...]] = (
    "ipfs_datasets_py.logic.ir_core",
    "ipfs_datasets_py.logic.formalization",
    "ipfs_datasets_py.logic.intent_ir",
    "ipfs_datasets_py.logic.legal_ir",
    "ipfs_datasets_py.logic.security_ir",
)

# Planned UI package (not yet present) + supervisor interface bridge
UI_DATASETS_PACKAGE: Final = "ipfs_datasets_py.logic.ui_ux_ir"
UI_BRIDGE_MODULES: Final[tuple[str, ...]] = (
    "ipfs_accelerate_py.agent_supervisor.proof.interface_contract_codegen",
)

CONSTRAINT_ADAPTER_MODULES: Final[Mapping[str, str]] = {
    "intent_ir": "ipfs_accelerate_py.agent_supervisor.proof.intent_constraint_adapter",
    "legal_ir": "ipfs_accelerate_py.agent_supervisor.proof.legal_constraint_adapter",
    "security_ir": "ipfs_accelerate_py.agent_supervisor.proof.security_constraint_adapter",
}

CONSTRAINT_ENTRYPOINTS: Final[Mapping[str, tuple[str, ...]]] = {
    "intent_ir": ("IntentConstraintAdapter", "compile_intent_constraints"),
    "legal_ir": ("LegalConstraintAdapter", "compile_legal_constraints"),
    "security_ir": (
        "SecurityConstraintAdapter",
        "compile_security_constraints",
        "evaluate_security_authorization",
    ),
}


class ScaIrIntegrationError(ValueError):
    """Malformed IR integration policy."""


@dataclass
class IrIntegrationPolicy:
    """Supervisor policy for IR family wiring into SCA repair/planning."""

    require_intent_ir: bool = True
    require_legal_ir: bool = True
    require_security_ir: bool = True
    require_ui_ir: bool = True  # package + bridge now available
    require_ui_interface_bridge: bool = True
    require_constraint_adapters: bool = True
    require_ir_constraint_compiler: bool = True
    require_datasets_packages: bool = True
    analysis_families: tuple[str, ...] = DEFAULT_IR_ANALYSIS_FAMILIES
    registry_families: tuple[str, ...] = DEFAULT_REGISTRY_IR_FAMILIES

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "IrIntegrationPolicy":
        raw = dict(raw or {})
        families = raw.get("analysisFamilies") or raw.get("analysis_families")
        registry = raw.get("registryFamilies") or raw.get("registry_families")
        return cls(
            require_intent_ir=bool(
                raw.get("requireIntentIr", raw.get("require_intent_ir", True))
            ),
            require_legal_ir=bool(
                raw.get("requireLegalIr", raw.get("require_legal_ir", True))
            ),
            require_security_ir=bool(
                raw.get(
                    "requireSecurityIr", raw.get("require_security_ir", True)
                )
            ),
            require_ui_ir=bool(
                raw.get("requireUiIr", raw.get("require_ui_ir", False))
            ),
            require_ui_interface_bridge=bool(
                raw.get(
                    "requireUiInterfaceBridge",
                    raw.get("require_ui_interface_bridge", True),
                )
            ),
            require_constraint_adapters=bool(
                raw.get(
                    "requireConstraintAdapters",
                    raw.get("require_constraint_adapters", True),
                )
            ),
            require_ir_constraint_compiler=bool(
                raw.get(
                    "requireIrConstraintCompiler",
                    raw.get("require_ir_constraint_compiler", True),
                )
            ),
            require_datasets_packages=bool(
                raw.get(
                    "requireDatasetsPackages",
                    raw.get("require_datasets_packages", True),
                )
            ),
            analysis_families=tuple(families or DEFAULT_IR_ANALYSIS_FAMILIES),
            registry_families=tuple(registry or DEFAULT_REGISTRY_IR_FAMILIES),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class IrSurfaceProbe:
    family: str
    available: bool
    role: str
    grants_execution_authority: bool = False
    module: str = ""
    detail: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "available": self.available,
            "role": self.role,
            "grants_execution_authority": self.grants_execution_authority,
            "module": self.module,
            "detail": self.detail,
            "error": self.error,
        }


def _try_import(module_path: str) -> tuple[Any | None, str]:
    try:
        return importlib.import_module(module_path), ""
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def probe_registry_adapters(
    families: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Discover IR adapter capabilities from the provider-free registry."""
    want = tuple(families or DEFAULT_REGISTRY_IR_FAMILIES)
    out: dict[str, Any] = {
        "available": False,
        "families": {},
        "discovered": [],
        "error": "",
    }
    try:
        from .proof.ir_adapters import IRAdapterRegistry
        from .proof.ir_registry import IRFamily, normalize_ir_family

        registry = IRAdapterRegistry(include_shared=True)
        caps = registry.discover_capabilities()
        discovered: list[dict[str, Any]] = []
        by_family: dict[str, dict[str, Any]] = {}
        for cap in caps:
            d = cap.to_dict() if hasattr(cap, "to_dict") else {}
            fam = str(
                d.get("family")
                or getattr(getattr(cap, "family", None), "value", "")
                or ""
            )
            entry = {
                "family": fam,
                "adapter_id": d.get("adapter_id"),
                "operations": list(d.get("operations") or []),
                "lazy": bool(d.get("lazy", True)),
                "grants_execution_authority": bool(
                    d.get("grants_execution_authority", False)
                ),
                "schema": d.get("schema"),
            }
            discovered.append(entry)
            by_family[fam] = entry

        # Ensure requested families are present (fail-closed gap report)
        for name in want:
            try:
                fam = normalize_ir_family(name)
                key = fam.value
            except Exception:  # noqa: BLE001
                key = name
            if key not in by_family:
                by_family[key] = {
                    "family": key,
                    "available": False,
                    "error": "adapter_not_registered",
                    "grants_execution_authority": False,
                }
            else:
                by_family[key]["available"] = True

        # Enum completeness for shared IR families
        enum_values = [item.value for item in IRFamily]
        out.update(
            {
                "available": True,
                "families": by_family,
                "discovered": discovered,
                "enum_families": enum_values,
                "requested": list(want),
                "ui_ir_in_registry_enum": "ui_ir" in enum_values,
            }
        )
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def probe_constraint_adapters() -> dict[str, Any]:
    """Import constraint adapters and confirm public compile entrypoints."""
    results: dict[str, Any] = {}
    for family, module_path in CONSTRAINT_ADAPTER_MODULES.items():
        mod, err = _try_import(module_path)
        if mod is None:
            results[family] = {
                "available": False,
                "module": module_path,
                "error": err,
            }
            continue
        missing = [
            name
            for name in CONSTRAINT_ENTRYPOINTS.get(family, ())
            if not hasattr(mod, name)
        ]
        results[family] = {
            "available": not missing,
            "module": module_path,
            "entrypoints": list(CONSTRAINT_ENTRYPOINTS.get(family, ())),
            "missing_entrypoints": missing,
            "grants_execution_authority": False,
        }
    return results


def probe_ir_constraint_compiler() -> dict[str, Any]:
    mod, err = _try_import(
        "ipfs_accelerate_py.agent_supervisor.proof.ir_constraint_compiler"
    )
    if mod is None:
        return {"available": False, "error": err}
    names = (
        "IRConstraintCompiler",
        "compile_plan_admission",
        "AdmissionDomain",
    )
    missing = [n for n in names if not hasattr(mod, n)]
    domains: list[str] = []
    try:
        AdmissionDomain = getattr(mod, "AdmissionDomain", None)
        if AdmissionDomain is not None:
            domains = [item.value for item in AdmissionDomain]
    except Exception as exc:  # noqa: BLE001
        return {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
            "missing": missing,
        }
    return {
        "available": not missing,
        "module": "ipfs_accelerate_py.agent_supervisor.proof.ir_constraint_compiler",
        "entrypoints": list(names),
        "missing": missing,
        "admission_domains": domains,
        "grants_execution_authority": False,
    }


def probe_datasets_ir_packages(
    packages: Sequence[str] | None = None,
) -> dict[str, Any]:
    packages = tuple(packages or DEFAULT_DATASETS_IR_PACKAGES)
    out: dict[str, Any] = {}
    for path in packages:
        short = path.rsplit(".", 1)[-1]
        mod, err = _try_import(path)
        out[short] = {
            "available": mod is not None,
            "module": path,
            "file": getattr(mod, "__file__", None) if mod is not None else None,
            "error": err,
        }
    return out


def probe_ui_ir_surface() -> dict[str, Any]:
    """UI IR surface: full package + interface bridge + IRFamily registration.

    Delegates to domain-agnostic :func:`proof.ir_integration.probe_ui_ir_surface`
    and preserves SCA-shaped bridge detail for existing reports.
    """
    bridge: dict[str, Any] = {}
    for path in UI_BRIDGE_MODULES:
        mod, err = _try_import(path)
        exports = []
        if mod is not None:
            for name in (
                "ActionContractCodegenConfig",
                "ActionContractSyncSpec",
                "ConfiguredActionContractSyncRunner",
                "PythonActionContractConfig",
                "JavaScriptActionContractConfig",
            ):
                if hasattr(mod, name):
                    exports.append(name)
        bridge[path.rsplit(".", 1)[-1]] = {
            "available": mod is not None,
            "module": path,
            "exports": exports,
            "error": err,
            "grants_execution_authority": False,
            "role": "interface_descriptor_codegen",
        }

    ux_mod, ux_err = _try_import(UI_DATASETS_PACKAGE)
    datasets_ui = {
        "available": ux_mod is not None,
        "module": UI_DATASETS_PACKAGE,
        "file": getattr(ux_mod, "__file__", None) if ux_mod is not None else None,
        "error": ux_err,
        "status": "available" if ux_mod is not None else "planned_unavailable",
        "grants_execution_authority": False,
        "role": "ui_ux_ir_document_family",
    }

    general: dict[str, Any] = {}
    try:
        from .proof.ir_integration import probe_ui_ir_surface as probe_general_ui

        general = probe_general_ui()
    except Exception as exc:  # noqa: BLE001
        general = {"error": f"{type(exc).__name__}: {exc}"}

    bridge_ok = all(v.get("available") for v in bridge.values())
    full_ok = bool(datasets_ui["available"]) or bool(
        general.get("full_ui_ux_ir_available")
    )
    return {
        "family": "ui_ir",
        "in_registry_enum": bool(general.get("in_registry_enum")),
        "registry_adapter": bool(general.get("registry_adapter")),
        "available": full_ok or bridge_ok,
        "full_ui_ux_ir_available": full_ok,
        "interface_bridge": bridge,
        "interface_bridge_available": bridge_ok,
        "datasets_ui_ux_ir": datasets_ui,
        "projections": general.get("projections") or {},
        "entrypoints": general.get("entrypoints") or {},
        "grants_execution_authority": False,
        "status": "full" if full_ok else ("bridge_only" if bridge_ok else "unavailable"),
        "notes": [
            "ui_ir is an IRFamily member (context-only authority).",
            "Python authority: ipfs_datasets_py.logic.ui_ux_ir.",
            "TypeScript peer: swissknife ui-ux-ir-codec / web / glasses adapters.",
            "Use interface_contract_codegen for UI action descriptors.",
            "Do not treat UI descriptors as kernel authority.",
        ],
    }


def probe_decision_ir_roots() -> dict[str, Any]:
    """Confirm decision contracts expose Intent/Legal/Security root accessors."""
    try:
        from .context import decision_contracts as dc

        kinds = []
        if hasattr(dc, "SemanticChangeKind"):
            kinds = [item.value for item in dc.SemanticChangeKind]
        root_attrs = [
            name
            for name in (
                "intent_ir_root",
                "legal_ir_root",
                "security_ir_root",
                "ui_ir_root",
                "interface_ir_root",
            )
            if hasattr(dc, name)
            or any(
                hasattr(cls, name)
                for cls in (getattr(dc, n, None) for n in dir(dc))
                if isinstance(cls, type)
            )
        ]
        # Property-style roots live on concrete classes — scan common types
        class_hits: dict[str, list[str]] = {}
        for cls_name in (
            "PinnedArtifactBundle",
            "DecisionContext",
            "WorkflowRoots",
        ):
            cls = getattr(dc, cls_name, None)
            if cls is None:
                continue
            hits = [
                attr
                for attr in (
                    "intent_ir_root",
                    "legal_ir_root",
                    "security_ir_root",
                    "ui_ir_root",
                )
                if hasattr(cls, attr)
            ]
            if hits:
                class_hits[cls_name] = hits

        # Broader scan for root property names
        for name in dir(dc):
            obj = getattr(dc, name)
            if not isinstance(obj, type):
                continue
            hits = [
                attr
                for attr in (
                    "intent_ir_root",
                    "legal_ir_root",
                    "security_ir_root",
                    "ui_ir_root",
                )
                if hasattr(obj, attr)
            ]
            if hits:
                class_hits[name] = hits

        return {
            "available": True,
            "semantic_change_kinds": kinds,
            "intent_legal_security_kinds_present": all(
                k in kinds for k in ("intent_ir", "legal_ir", "security_ir")
            ),
            "ui_kind_present": "ui_ir" in kinds or "interface_ir" in kinds,
            "root_bearing_classes": class_hits,
            "root_attrs_detected": sorted(
                {a for hits in class_hits.values() for a in hits}
            ),
        }
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def probe_ir_integration(
    policy: IrIntegrationPolicy | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Full IR integration inventory for SCA symbolic repair / planning.

    Delegates shared + structural IR inventory to the domain-agnostic
    :mod:`proof.ir_integration` module, then layers SCA-specific decision-root
    probes and report schema. SCA is one consumer — not the only domain.
    """
    if isinstance(policy, Mapping):
        policy = IrIntegrationPolicy.from_mapping(policy)
    elif policy is None:
        policy = IrIntegrationPolicy()

    # Domain-agnostic inventory (intent/legal/security/ui + AST/KG/vector + hooks)
    try:
        from .proof.ir_integration import (
            IrIntegrationPolicy as GeneralIrPolicy,
            probe_ir_integration as probe_general,
        )

        general_policy = GeneralIrPolicy.from_mapping(
            {
                **policy.to_dict(),
                "domain": "sca",
                # SCA historically required datasets packages for shared IR
                "require_datasets_packages": policy.require_datasets_packages,
                "require_ast": True,
                "require_knowledge_graph": True,
                "require_vector_index": True,
                "require_live_apply": True,
                "analysis_families": list(policy.analysis_families)
                + ["ast", "knowledge_graph", "vector_index"],
            }
        )
        # de-dupe analysis families while preserving order
        seen: set[str] = set()
        fams: list[str] = []
        for f in general_policy.analysis_families:
            if f not in seen:
                seen.add(f)
                fams.append(f)
        general_policy.analysis_families = tuple(fams)
        general = probe_general(general_policy, domain="sca")
    except Exception as exc:  # noqa: BLE001
        general = {
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
            "families": {},
            "gates": {},
        }

    # SCA-local probes (compat with existing report shape)
    registry = probe_registry_adapters(policy.registry_families)
    constraints = probe_constraint_adapters()
    compiler = probe_ir_constraint_compiler()
    datasets = probe_datasets_ir_packages()
    ui = probe_ui_ir_surface()
    decision = probe_decision_ir_roots()

    families_status: dict[str, dict[str, Any]] = dict(general.get("families") or {})
    # Ensure classic SCA keys remain if general probe failed partially
    for fam in ("intent_ir", "legal_ir", "security_ir"):
        if fam in families_status:
            continue
        reg_ok = bool((registry.get("families") or {}).get(fam, {}).get("available"))
        con_ok = bool((constraints.get(fam) or {}).get("available"))
        pkg_ok = bool((datasets.get(fam) or {}).get("available"))
        families_status[fam] = {
            "available": reg_ok
            and con_ok
            and (pkg_ok or not policy.require_datasets_packages),
            "registry_adapter": reg_ok,
            "constraint_adapter": con_ok,
            "datasets_package": pkg_ok,
            "grants_execution_authority": False,
            "role": {
                "intent_ir": "required_work_constraints",
                "legal_ir": "applicability_constraints",
                "security_ir": "authorization_decision_inputs",
            }[fam],
        }

    if "ui_ir" not in families_status:
        families_status["ui_ir"] = {
            "available": bool(ui.get("available")),
            "registry_adapter": bool(ui.get("in_registry_enum")),
            "constraint_adapter": False,
            "datasets_package": bool(ui.get("full_ui_ux_ir_available")),
            "interface_bridge": bool(ui.get("available")),
            "grants_execution_authority": False,
            "role": "interface_descriptor_surface",
            "status": (
                "bridge_only"
                if ui.get("available") and not ui.get("full_ui_ux_ir_available")
                else ("full" if ui.get("full_ui_ux_ir_available") else "unavailable")
            ),
        }

    gates = {
        "intent_ir": (
            not policy.require_intent_ir
            or bool(families_status.get("intent_ir", {}).get("available"))
        ),
        "legal_ir": (
            not policy.require_legal_ir
            or bool(families_status.get("legal_ir", {}).get("available"))
        ),
        "security_ir": (
            not policy.require_security_ir
            or bool(families_status.get("security_ir", {}).get("available"))
        ),
        "ui_ir": (
            not policy.require_ui_ir
            or bool(families_status.get("ui_ir", {}).get("available"))
        ),
        "ui_interface_bridge": (
            not policy.require_ui_interface_bridge
            or bool(ui.get("available"))
        ),
        "ast": bool(families_status.get("ast", {}).get("available", True)),
        "knowledge_graph": bool(
            families_status.get("knowledge_graph", {}).get("available", True)
        ),
        "vector_index": bool(
            families_status.get("vector_index", {}).get("available", True)
        ),
        "constraint_adapters": (
            not policy.require_constraint_adapters
            or all(
                (constraints.get(f) or {}).get("available")
                for f in ("intent_ir", "legal_ir", "security_ir")
            )
        ),
        "ir_constraint_compiler": (
            not policy.require_ir_constraint_compiler
            or bool(compiler.get("available"))
        ),
        "registry_adapters": bool(registry.get("available")),
        "no_false_execution_grants": all(
            not bool(v.get("grants_execution_authority"))
            for v in families_status.values()
        ),
        "general_ir_integration": bool(general.get("passed", True)),
    }
    passed = all(gates.values())

    return {
        "schema": SCA_IR_INTEGRATION_REPORT_SCHEMA,
        "interface": SCA_IR_INTEGRATION_INTERFACE,
        "domain": "sca",
        "passed": passed,
        "policy": policy.to_dict(),
        "gates": gates,
        "families": families_status,
        "analysis_families": list(
            dict.fromkeys(
                list(policy.analysis_families)
                + ["ast", "knowledge_graph", "vector_index"]
            )
        ),
        "registry": registry,
        "constraint_adapters": constraints,
        "ir_constraint_compiler": compiler,
        "datasets_packages": datasets,
        "ui_ir": ui,
        "decision_contracts": decision,
        "structural": general.get("structural"),
        "consumers": general.get("consumers"),
        "live_apply": general.get("live_apply"),
        "general_integration": {
            "schema": general.get("schema"),
            "interface": general.get("interface"),
            "passed": general.get("passed"),
            "domain": general.get("domain"),
        },
        "notes": [
            "SCA is one consumer of the domain-agnostic IR stack (proof.ir_integration).",
            "Intent/Legal/Security IR adapters + constraint compilers are wired.",
            "UI IR uses interface_contract_codegen bridge until ui_ux_ir lands.",
            "AST / knowledge_graph / vector_index are structural intermediate IRs.",
            "Planner/doctor/symbolic-repair hooks apply IR logic live (not board-only).",
            "No IR surface grants execution authority from inventory alone.",
            "Security authorization evaluation stays fail-closed without grants.",
            "KERNEL_VERIFIED remains observation-bound; IR wiring is context only.",
        ],
    }


def load_ir_policy_from_supervisor_profile(
    profile_path: str | None = None,
) -> IrIntegrationPolicy:
    """Load ``irIntegrationPolicy`` (or nested symbolicRepairPolicy.irIntegration)."""
    import json
    from pathlib import Path

    if profile_path is None:
        here = Path(__file__).resolve()
        candidates = [
            here.parents[4]
            / "config"
            / "swissknife_symbolic_contract_assurance_supervisor.json",
            Path.cwd()
            / "config"
            / "swissknife_symbolic_contract_assurance_supervisor.json",
        ]
        for c in candidates:
            if c.is_file():
                profile_path = str(c)
                break
    if not profile_path or not Path(profile_path).is_file():
        return IrIntegrationPolicy()
    doc = json.loads(Path(profile_path).read_text(encoding="utf-8"))
    raw = (
        doc.get("irIntegrationPolicy")
        or doc.get("ir_integration_policy")
        or (doc.get("symbolicRepairPolicy") or {}).get("irIntegration")
        or (doc.get("symbolicRepairPolicy") or {}).get("ir_integration")
        or {}
    )
    return IrIntegrationPolicy.from_mapping(raw)


__all__ = [
    "CONSTRAINT_ADAPTER_MODULES",
    "DEFAULT_DATASETS_IR_PACKAGES",
    "DEFAULT_IR_ANALYSIS_FAMILIES",
    "DEFAULT_REGISTRY_IR_FAMILIES",
    "SCA_IR_INTEGRATION_INTERFACE",
    "SCA_IR_INTEGRATION_REPORT_SCHEMA",
    "UI_BRIDGE_MODULES",
    "UI_DATASETS_PACKAGE",
    "IrIntegrationPolicy",
    "IrSurfaceProbe",
    "ScaIrIntegrationError",
    "load_ir_policy_from_supervisor_profile",
    "probe_constraint_adapters",
    "probe_datasets_ir_packages",
    "probe_decision_ir_roots",
    "probe_ir_constraint_compiler",
    "probe_ir_integration",
    "probe_registry_adapters",
    "probe_ui_ir_surface",
]
