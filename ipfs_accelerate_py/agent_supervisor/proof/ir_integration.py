"""Domain-agnostic IR integration inventory for the agent supervisor.

**Not SCA-taskboard-specific.** Any consumer (planner, doctor, symbolic
repair, contract repair, autonomous repair, SCA) uses this inventory to
discover and gate:

Shared IR
  ``intent_ir`` · ``legal_ir`` · ``security_ir`` · ``ui_ir``

Structural IR
  ``ast`` · ``knowledge_graph`` · ``vector_index``

Authority rules (fail-closed):

* Intent describes required work; never authorizes execution.
* Legal applicability is context; never a capability grant.
* Security evaluation alone may PERMIT/DENY/UNKNOWN.
* UI / AST / KG / vector hits are non-authoritative intermediate context.
* KERNEL_VERIFIED remains observation-bound outside this module.
"""

from __future__ import annotations

import importlib
import hashlib
from collections.abc import Mapping as MappingABC, Sequence as SequenceABC
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

from .ir_logic_application import (
    DEFAULT_APPLY_FAMILIES,
    SHARED_IR_FAMILIES,
    STRUCTURAL_IR_FAMILIES,
    IrLogicApplyPolicy,
    IrWorkSurface,
    apply_logic_to_ir,
)


IR_INTEGRATION_INTERFACE: Final = "IrIntegration@1"
IR_INTEGRATION_REPORT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/ir-integration@1"
)

# Registry-backed shared-IR families (authoritative enum; ui_ir is bridge-only)
DEFAULT_REGISTRY_IR_FAMILIES: Final[tuple[str, ...]] = (
    "ir_core",
    "formalization",
    "intent_ir",
    "legal_ir",
    "security_ir",
    "ui_ir",
)

DEFAULT_DATASETS_IR_PACKAGES: Final[tuple[str, ...]] = (
    "ipfs_datasets_py.logic.ir_core",
    "ipfs_datasets_py.logic.formalization",
    "ipfs_datasets_py.logic.intent_ir",
    "ipfs_datasets_py.logic.legal_ir",
    "ipfs_datasets_py.logic.security_ir",
)

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

STRUCTURAL_MODULES: Final[Mapping[str, tuple[str, ...]]] = {
    "ast": (
        "ipfs_accelerate_py.agent_supervisor.analysis.analysis_ast_index",
        "ipfs_accelerate_py.agent_supervisor.proof.ir_structural_application",
    ),
    "knowledge_graph": (
        "ipfs_accelerate_py.agent_supervisor.analysis.semantic_dependency_graph",
        "ipfs_accelerate_py.agent_supervisor.proof.ir_structural_application",
    ),
    "vector_index": (
        "ipfs_accelerate_py.agent_supervisor.analysis.code_symbol_vector_index",
        "ipfs_accelerate_py.agent_supervisor.proof.ir_structural_application",
    ),
}

CONSUMER_HOOK_MODULES: Final[Mapping[str, str]] = {
    "planner": "ipfs_accelerate_py.agent_supervisor.planning.ir_logic_hooks",
    "doctor": "ipfs_accelerate_py.agent_supervisor.planning.ir_logic_consumers",
    "symbolic_repair": "ipfs_accelerate_py.agent_supervisor.proof.ir_logic_application",
    "formal_plan_compiler": "ipfs_accelerate_py.agent_supervisor.planning.formal_plan_compiler",
    "autonomous_repair": "ipfs_accelerate_py.agent_supervisor.autonomous_repair.engine",
}


class IrIntegrationError(ValueError):
    """Malformed IR integration policy."""


@dataclass
class IrIntegrationPolicy:
    """Policy for IR family wiring into any supervisor domain."""

    domain: str = "agent_supervisor"
    require_intent_ir: bool = True
    require_legal_ir: bool = True
    require_security_ir: bool = True
    require_ui_ir: bool = False  # package exists; keep optional for legacy profiles
    require_ui_interface_bridge: bool = True
    require_ast: bool = True
    require_knowledge_graph: bool = True
    require_vector_index: bool = True
    require_constraint_adapters: bool = True
    require_ir_constraint_compiler: bool = True
    require_datasets_packages: bool = False  # fixtures work without live packages
    require_live_apply: bool = True
    analysis_families: tuple[str, ...] = DEFAULT_APPLY_FAMILIES
    registry_families: tuple[str, ...] = DEFAULT_REGISTRY_IR_FAMILIES

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "IrIntegrationPolicy":
        raw = dict(raw or {})
        families = raw.get("analysisFamilies") or raw.get("analysis_families")
        registry = raw.get("registryFamilies") or raw.get("registry_families")
        return cls(
            domain=str(raw.get("domain") or "agent_supervisor"),
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
            require_ast=bool(raw.get("requireAst", raw.get("require_ast", True))),
            require_knowledge_graph=bool(
                raw.get(
                    "requireKnowledgeGraph",
                    raw.get("require_knowledge_graph", True),
                )
            ),
            require_vector_index=bool(
                raw.get(
                    "requireVectorIndex",
                    raw.get("require_vector_index", True),
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
                    raw.get("require_datasets_packages", False),
                )
            ),
            require_live_apply=bool(
                raw.get("requireLiveApply", raw.get("require_live_apply", True))
            ),
            analysis_families=tuple(families or DEFAULT_APPLY_FAMILIES),
            registry_families=tuple(registry or DEFAULT_REGISTRY_IR_FAMILIES),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
        from .ir_adapters import IRAdapterRegistry
        from .ir_registry import IRFamily, normalize_ir_family

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
    if hasattr(mod, "AdmissionDomain"):
        try:
            domains = [d.value for d in mod.AdmissionDomain]
        except Exception:  # noqa: BLE001
            domains = []
    return {
        "available": not missing,
        "module": mod.__name__,
        "missing_entrypoints": missing,
        "admission_domains": domains,
        "grants_execution_authority": False,
    }


def probe_datasets_ir_packages() -> dict[str, Any]:
    results: dict[str, Any] = {}
    for pkg in DEFAULT_DATASETS_IR_PACKAGES:
        short = pkg.rsplit(".", 1)[-1]
        mod, err = _try_import(pkg)
        results[short] = {
            "available": mod is not None,
            "package": pkg,
            "error": err,
        }
    return results


def probe_ui_ir_surface() -> dict[str, Any]:
    """UI IR: full package + interface bridge + TypeScript peer inventory."""
    bridge_ok = True
    bridge_errors: list[str] = []
    for mod_path in UI_BRIDGE_MODULES:
        mod, err = _try_import(mod_path)
        if mod is None:
            bridge_ok = False
            bridge_errors.append(f"{mod_path}: {err}")
    full_mod, full_err = _try_import(UI_DATASETS_PACKAGE)
    entrypoints: dict[str, bool] = {}
    if full_mod is not None:
        for name in (
            "decode_ui_ir",
            "canonicalize_ui_ir",
            "ui_ir_sha256",
            "migrate_ui_ir",
            "decode_ui_ir_with_migration",
            "UIIRDocument",
        ):
            entrypoints[name] = hasattr(full_mod, name)

    enum_has_ui = False
    adapter_ok = False
    try:
        from .ir_registry import IRFamily
        from .ir_adapters import IRAdapterRegistry

        enum_has_ui = "ui_ir" in [i.value for i in IRFamily]
        registry = IRAdapterRegistry(include_shared=True)
        _caps = {
            str(getattr(getattr(c, "family", None), "value", c)): c
            for c in registry.discover_capabilities()
        }
        # discover returns capability objects
        adapter_ok = any(
            str(getattr(getattr(c, "family", None), "value", "")) == "ui_ir"
            or (isinstance(c, Mapping) and c.get("family") == "ui_ir")
            or (
                hasattr(c, "to_dict")
                and str((c.to_dict() or {}).get("family") or "") == "ui_ir"
            )
            for c in registry.discover_capabilities()
        )
        if not adapter_ok:
            for c in registry.discover_capabilities():
                d = c.to_dict() if hasattr(c, "to_dict") else {}
                if str(d.get("family") or "") == "ui_ir":
                    adapter_ok = True
                    break
    except Exception:  # noqa: BLE001
        pass

    projections: dict[str, Any] = {}
    try:
        from ipfs_datasets_py.logic.ui_ux_ir.projections import (
            inventory_projection_capabilities,
        )

        projections = inventory_projection_capabilities()
    except Exception as exc:  # noqa: BLE001
        projections = {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    full_ok = full_mod is not None and all(entrypoints.values())
    return {
        "family": "ui_ir",
        "available": full_ok or bridge_ok,
        "in_registry_enum": enum_has_ui,
        "registry_adapter": adapter_ok,
        "full_ui_ux_ir_available": full_ok,
        "interface_bridge_available": bridge_ok,
        "interface_bridge_modules": list(UI_BRIDGE_MODULES),
        "datasets_package": UI_DATASETS_PACKAGE,
        "datasets_error": full_err,
        "entrypoints": entrypoints,
        "bridge_errors": bridge_errors,
        "projections": projections,
        "grants_execution_authority": False,
        "role": "interface_descriptor_surface",
        "status": "full" if full_ok else ("bridge_only" if bridge_ok else "unavailable"),
        "notes": [
            "ui_ir is an IRFamily member (context-only authority).",
            "Python authority: ipfs_datasets_py.logic.ui_ux_ir.",
            "TypeScript peer codec/renderers live under swissknife/src/services/.",
            "Interface bridge still projects action contracts for mediation.",
            "No UI IR surface grants execution authority.",
        ],
    }


def probe_structural_ir_surfaces(
    *,
    domain: str = "agent_supervisor",
    operation: str = "ir.integration.probe",
) -> dict[str, Any]:
    """Probe AST / knowledge-graph / vector-index structural IR applicability."""
    out: dict[str, Any] = {
        "available": False,
        "families": {},
        "modules": {},
        "live_apply": {},
        "error": "",
    }
    module_ok = True
    for fam, mods in STRUCTURAL_MODULES.items():
        fam_mods: dict[str, Any] = {}
        for m in mods:
            mod, err = _try_import(m)
            fam_mods[m] = {"available": mod is not None, "error": err}
            if mod is None:
                module_ok = False
        out["modules"][fam] = fam_mods

    try:
        from .ir_structural_application import apply_structural_logic

        structural = apply_structural_logic(
            operation=operation,
            contract_id=f"probe:{domain}",
            finding_kind="integration_probe",
            domain=domain,
            families=STRUCTURAL_IR_FAMILIES,
        )
        fams = structural.get("families") or {}
        for name in STRUCTURAL_IR_FAMILIES:
            doc = fams.get(name) or {}
            out["families"][name] = {
                "available": bool(doc.get("ok") or doc.get("available")),
                "ok": bool(doc.get("ok")),
                "status": doc.get("status"),
                "role": doc.get("role"),
                "logic_applied": list(doc.get("logic_applied") or []),
                "grants_execution_authority": False,
            }
            out["live_apply"][name] = {
                "ok": bool(doc.get("ok")),
                "status": doc.get("status"),
            }
        out["available"] = module_ok and all(
            (out["families"].get(n) or {}).get("available")
            for n in STRUCTURAL_IR_FAMILIES
        )
        out["errors"] = list(structural.get("errors") or [])
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
        for name in STRUCTURAL_IR_FAMILIES:
            out["families"].setdefault(
                name,
                {
                    "available": False,
                    "ok": False,
                    "error": out["error"],
                    "grants_execution_authority": False,
                },
            )
    return out


def probe_consumer_hooks() -> dict[str, Any]:
    """Confirm planner / doctor / repair / compiler IR hook modules import."""
    results: dict[str, Any] = {}
    for consumer, module_path in CONSUMER_HOOK_MODULES.items():
        mod, err = _try_import(module_path)
        results[consumer] = {
            "available": mod is not None,
            "module": module_path,
            "error": err,
            "grants_execution_authority": False,
        }
    # Specific entrypoints for hooks
    hooks_mod, _ = _try_import(CONSUMER_HOOK_MODULES["planner"])
    if hooks_mod is not None:
        for name in (
            "prepare_planning_context",
            "inject_ir_into_formal_plan_source",
            "attach_ir_logic_to_doctor_request",
            "compose_hard_gate_with_ir",
        ):
            results["planner"][f"has_{name}"] = hasattr(hooks_mod, name)
    consumers_mod, _ = _try_import(CONSUMER_HOOK_MODULES["doctor"])
    if consumers_mod is not None:
        for name in (
            "diagnose_with_ir_logic",
            "enrich_planning_context_with_ir_logic",
            "attach_ir_logic_to_symbolic_plan",
        ):
            results["doctor"][f"has_{name}"] = hasattr(consumers_mod, name)
    return results


def probe_live_apply(
    *,
    domain: str = "agent_supervisor",
    operation: str = "ir.integration.probe",
    families: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run a full multi-family apply on a synthetic work surface."""
    try:
        doc = apply_logic_to_ir(
            IrWorkSurface(
                operation=operation,
                kind="integration_probe",
                contract_id=f"probe:{domain}:{operation}",
                domain=domain,
                consumer="ir_integration",
            ),
            policy=IrLogicApplyPolicy(
                families=tuple(families or DEFAULT_APPLY_FAMILIES),
                evaluate_security=True,
                include_plan_admission=False,
            ),
            domain=domain,
            consumer="ir_integration",
        )
        return {
            "available": True,
            "passed": bool(doc.get("passed")),
            "family_ok": dict(doc.get("family_ok") or {}),
            "gates": dict(doc.get("gates") or {}),
            "families": {
                name: {
                    "ok": (fdoc or {}).get("ok"),
                    "status": (fdoc or {}).get("status"),
                    "logic_applied": list((fdoc or {}).get("logic_applied") or []),
                }
                for name, fdoc in (doc.get("families") or {}).items()
            },
            "grants_execution_authority": False,
            "domain": domain,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "available": False,
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
            "grants_execution_authority": False,
            "domain": domain,
        }


def probe_ir_integration(
    policy: IrIntegrationPolicy | Mapping[str, Any] | None = None,
    *,
    domain: str | None = None,
) -> dict[str, Any]:
    """Full IR integration inventory for any supervisor domain.

    Includes shared IR (intent/legal/security/ui), structural IR (AST/KG/vector),
    consumer hooks (planner/doctor/symbolic repair), and a live apply sample.
    """
    if isinstance(policy, Mapping):
        policy = IrIntegrationPolicy.from_mapping(policy)
    elif policy is None:
        policy = IrIntegrationPolicy()
    if domain:
        policy.domain = domain
    domain = policy.domain or "agent_supervisor"

    registry = probe_registry_adapters(policy.registry_families)
    constraints = probe_constraint_adapters()
    compiler = probe_ir_constraint_compiler()
    datasets = probe_datasets_ir_packages()
    ui = probe_ui_ir_surface()
    structural = probe_structural_ir_surfaces(domain=domain)
    consumers = probe_consumer_hooks()
    live = probe_live_apply(
        domain=domain,
        families=policy.analysis_families,
    )

    families_status: dict[str, dict[str, Any]] = {}
    for fam in ("intent_ir", "legal_ir", "security_ir"):
        reg_ok = bool((registry.get("families") or {}).get(fam, {}).get("available"))
        con_ok = bool((constraints.get(fam) or {}).get("available"))
        pkg_ok = bool((datasets.get(fam) or {}).get("available"))
        live_ok = bool((live.get("family_ok") or {}).get(fam))
        families_status[fam] = {
            "available": reg_ok
            and con_ok
            and (pkg_ok or not policy.require_datasets_packages)
            and (live_ok or not policy.require_live_apply),
            "registry_adapter": reg_ok,
            "constraint_adapter": con_ok,
            "datasets_package": pkg_ok,
            "live_apply": live_ok,
            "grants_execution_authority": False,
            "role": {
                "intent_ir": "required_work_constraints",
                "legal_ir": "applicability_constraints",
                "security_ir": "authorization_decision_inputs",
            }[fam],
        }

    families_status["ui_ir"] = {
        "available": bool(ui.get("available"))
        and (bool((live.get("family_ok") or {}).get("ui_ir")) or not policy.require_live_apply),
        "registry_adapter": bool(ui.get("in_registry_enum")),
        "constraint_adapter": False,
        "datasets_package": bool(ui.get("full_ui_ux_ir_available")),
        "interface_bridge": bool(ui.get("available")),
        "live_apply": bool((live.get("family_ok") or {}).get("ui_ir")),
        "grants_execution_authority": False,
        "role": "interface_descriptor_surface",
        "status": (
            "bridge_only"
            if ui.get("available") and not ui.get("full_ui_ux_ir_available")
            else ("full" if ui.get("full_ui_ux_ir_available") else "unavailable")
        ),
    }

    for fam in STRUCTURAL_IR_FAMILIES:
        sdoc = (structural.get("families") or {}).get(fam) or {}
        live_ok = bool((live.get("family_ok") or {}).get(fam))
        families_status[fam] = {
            "available": bool(sdoc.get("available"))
            and (live_ok or not policy.require_live_apply),
            "live_apply": live_ok,
            "status": sdoc.get("status"),
            "role": sdoc.get("role")
            or f"{fam}_intermediate_representation",
            "logic_applied": list(sdoc.get("logic_applied") or []),
            "grants_execution_authority": False,
        }

    gates = {
        "intent_ir": (
            not policy.require_intent_ir
            or bool(families_status["intent_ir"]["available"])
        ),
        "legal_ir": (
            not policy.require_legal_ir
            or bool(families_status["legal_ir"]["available"])
        ),
        "security_ir": (
            not policy.require_security_ir
            or bool(families_status["security_ir"]["available"])
        ),
        "ui_ir": (
            not policy.require_ui_ir
            or bool(families_status["ui_ir"]["available"])
        ),
        "ui_interface_bridge": (
            not policy.require_ui_interface_bridge
            or bool(ui.get("available"))
        ),
        "ast": (
            not policy.require_ast
            or bool(families_status["ast"]["available"])
        ),
        "knowledge_graph": (
            not policy.require_knowledge_graph
            or bool(families_status["knowledge_graph"]["available"])
        ),
        "vector_index": (
            not policy.require_vector_index
            or bool(families_status["vector_index"]["available"])
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
        "live_apply": (
            not policy.require_live_apply or bool(live.get("passed"))
        ),
        "consumer_hooks": all(
            (consumers.get(c) or {}).get("available")
            for c in ("planner", "doctor", "symbolic_repair")
        ),
        "no_false_execution_grants": all(
            not bool(v.get("grants_execution_authority"))
            for v in families_status.values()
        ),
    }
    passed = all(gates.values())

    return {
        "schema": IR_INTEGRATION_REPORT_SCHEMA,
        "interface": IR_INTEGRATION_INTERFACE,
        "domain": domain,
        "passed": passed,
        "policy": policy.to_dict(),
        "gates": gates,
        "families": families_status,
        "analysis_families": list(policy.analysis_families),
        "shared_ir_families": list(SHARED_IR_FAMILIES),
        "structural_ir_families": list(STRUCTURAL_IR_FAMILIES),
        "registry": registry,
        "constraint_adapters": constraints,
        "ir_constraint_compiler": compiler,
        "datasets_packages": datasets,
        "ui_ir": ui,
        "structural": structural,
        "consumers": consumers,
        "live_apply": live,
        "notes": [
            "Domain-agnostic IR stack: not bound to the SCA taskboard.",
            "Intent/Legal/Security adapters + constraint compilers are wired.",
            "UI IR: full package ipfs_datasets_py.logic.ui_ux_ir + interface_contract bridge.",
            "AST/KG/vector are structural intermediate representations.",
            "Planner/doctor/formal-plan/symbolic-repair hooks apply IR logic live.",
            "No IR surface grants execution authority from inventory alone.",
            "KERNEL_VERIFIED remains observation-bound; IR wiring is context only.",
        ],
    }


def load_ir_policy_from_mapping(
    raw: Mapping[str, Any] | None = None,
) -> IrIntegrationPolicy:
    """Load policy from a nested supervisor profile fragment."""
    raw = dict(raw or {})
    nested = (
        raw.get("irIntegrationPolicy")
        or raw.get("ir_integration_policy")
        or raw.get("irIntegration")
        or raw.get("ir_integration")
        or {}
    )
    if isinstance(nested, Mapping) and nested:
        return IrIntegrationPolicy.from_mapping(nested)
    return IrIntegrationPolicy.from_mapping(raw)


# DCR-030 deliberately has a smaller authority surface than the historical
# inventory above.  It invokes one current, dependency-light datasets API only
# after DCR-024 findings and DCR-004 content-bound capability evidence agree.
DATASETS_LOGIC_IR_FACADE_INTERFACE: Final = "DatasetsLogicIrFacade@1"
DATASETS_LOGIC_IR_INPUT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/datasets-logic-ir-input@1"
)
DATASETS_LOGIC_IR_RESULT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/datasets-logic-ir-result@1"
)
DATASETS_LOGIC_IDENTITY_MODULE: Final = "ipfs_datasets_py.logic.ir_core.identity"
_REQUIRED_CAPABILITY_EVIDENCE_KINDS: Final[tuple[str, ...]] = (
    "initialization",
    "reconstruction",
    "self_test",
)


class DatasetsLogicIrDisposition(str, Enum):
    """Closed DCR-030 results; none grants mutation authority."""

    NORMALIZED = "normalized_candidate"
    INTEGRATION_PENDING = "integration_pending"
    UNAVAILABLE = "unavailable"
    UNSUPPORTED = "unsupported"


class DatasetsLogicIrEvidenceKind(str, Enum):
    """The sole DCR-030 evidence classes accepted by the integration facade."""

    SOURCE_BYTES = "source_bytes"
    FOREST = "forest"
    GRAPH = "graph"
    FINDING = "finding"


def _compact_identifier(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field} must be non-empty exact text")
    if any(character.isspace() for character in value):
        raise ValueError(f"{field} must not contain whitespace")
    return value


@dataclass(frozen=True)
class DatasetsLogicIrEvidence:
    """Typed input evidence, bound to an external CID and local bytes digest."""

    kind: DatasetsLogicIrEvidenceKind | str
    cid: str
    payload: bytes | Mapping[str, Any]
    content_sha256: str = ""

    def __post_init__(self) -> None:
        try:
            kind = DatasetsLogicIrEvidenceKind(self.kind)
        except (TypeError, ValueError) as exc:
            raise ValueError("unsupported DCR-030 evidence kind") from exc
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "cid", _compact_identifier(self.cid, field="cid"))
        if kind is DatasetsLogicIrEvidenceKind.SOURCE_BYTES:
            if not isinstance(self.payload, bytes):
                raise ValueError("source_bytes evidence must contain bytes")
            encoded = self.payload
        else:
            if not isinstance(self.payload, MappingABC):
                raise ValueError(f"{kind.value} evidence must contain a mapping")
            from .formal_verification_contracts import canonical_json_bytes

            encoded = canonical_json_bytes(dict(self.payload))
        digest = "sha256:" + hashlib.sha256(encoded).hexdigest()
        if self.content_sha256 and self.content_sha256 != digest:
            raise ValueError("evidence content_sha256 does not bind its payload")
        object.__setattr__(self, "content_sha256", digest)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "cid": self.cid,
            "content_sha256": self.content_sha256,
            "kind": self.kind.value,
        }
        if self.kind is DatasetsLogicIrEvidenceKind.SOURCE_BYTES:
            payload["byte_length"] = len(self.payload)
        else:
            payload["payload"] = dict(self.payload)
        return payload


@dataclass(frozen=True)
class DatasetsLogicIrResult:
    """A deterministic, non-authoritative DCR-030 normalization result."""

    disposition: DatasetsLogicIrDisposition
    reason_codes: tuple[str, ...]
    input_cids: tuple[str, ...]
    module_binding: Mapping[str, str] = field(default_factory=dict)
    normalized_ir: Mapping[str, Any] = field(default_factory=dict)

    @property
    def mutation_authorized(self) -> bool:
        return False

    @property
    def model_call_count(self) -> int:
        return 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "interface": DATASETS_LOGIC_IR_FACADE_INTERFACE,
            "schema": DATASETS_LOGIC_IR_RESULT_SCHEMA,
            "disposition": self.disposition.value,
            "input_cids": list(self.input_cids),
            "model_call_count": 0,
            "mutation_authorized": False,
            "module_binding": dict(self.module_binding),
            "normalized_ir": dict(self.normalized_ir),
            "reason_codes": list(self.reason_codes),
        }


def _unavailable_result(
    disposition: DatasetsLogicIrDisposition,
    *reason_codes: str,
    evidence: Sequence[DatasetsLogicIrEvidence] = (),
    module_binding: Mapping[str, str] | None = None,
) -> DatasetsLogicIrResult:
    return DatasetsLogicIrResult(
        disposition=disposition,
        reason_codes=tuple(sorted(set(reason_codes))),
        input_cids=tuple(sorted(item.cid for item in evidence)),
        module_binding=dict(module_binding or {}),
    )


def _verify_dcr024_findings(
    finding: DatasetsLogicIrEvidence,
    *,
    graph: DatasetsLogicIrEvidence,
    forest: DatasetsLogicIrEvidence,
) -> tuple[bool, str]:
    """Accept only a current, ready DCR-024 report bound to these graph roots."""

    from ..analysis.mcp_contract_mismatch import (
        MCP_CONTRACT_MISMATCH_INTERFACE,
        MCP_CONTRACT_MISMATCH_SCHEMA,
    )
    from .formal_verification_contracts import content_identity

    raw = finding.payload
    if not isinstance(raw, MappingABC):  # Defensive; evidence validates this already.
        return False, "dcr024_findings_not_mapping"
    report = dict(raw)
    stored_cid = report.pop("findings_cid", "")
    if (
        report.get("schema") != MCP_CONTRACT_MISMATCH_SCHEMA
        or report.get("interface") != MCP_CONTRACT_MISMATCH_INTERFACE
        or report.get("authoritative") is not False
        or stored_cid != content_identity(report)
        or finding.cid != stored_cid
    ):
        return False, "dcr024_findings_identity_or_schema_invalid"
    if (
        report.get("dcr023_current_valid") is not True
        or report.get("production_readiness") != "ready"
        or report.get("findings") != []
    ):
        return False, "dcr024_findings_not_current_ready"
    snapshot_roots = report.get("snapshot_roots")
    if not isinstance(snapshot_roots, MappingABC):
        return False, "dcr024_snapshot_roots_invalid"
    if report.get("graph_cid") != graph.cid or snapshot_roots.get("forest") != forest.cid:
        return False, "dcr024_roots_do_not_bind_ir_inputs"
    return True, stored_cid


def _verify_source_evidence(source: DatasetsLogicIrEvidence) -> tuple[bool, str]:
    """Require a locally recomputed source-byte content identity."""

    from .formal_verification_contracts import content_identity

    expected_cid = content_identity(
        {
            "schema": DATASETS_LOGIC_IR_INPUT_SCHEMA + "/source-bytes",
            "sha256": source.content_sha256,
        }
    )
    if source.cid != expected_cid:
        return False, "source_bytes_cid_or_digest_invalid"
    return True, ""


def _verify_dcr011_forest(forest: DatasetsLogicIrEvidence) -> tuple[bool, str]:
    """Require an exact DCR-011 portable forest identity, never a label."""

    from ..analysis.deterministic_repair_forest import (
        DCR_FOREST_PORTABLE_SCHEMA,
        DCR_FOREST_SCHEMA,
    )
    from .formal_verification_contracts import content_identity

    value = forest.payload
    if not isinstance(value, MappingABC):
        return False, "dcr011_forest_not_mapping"
    portable = value.get("portable")
    if (
        value.get("schema") != DCR_FOREST_SCHEMA
        or value.get("interface") != "DeterministicRepairForest@1"
        or not isinstance(portable, MappingABC)
        or portable.get("schema") != DCR_FOREST_PORTABLE_SCHEMA
    ):
        return False, "dcr011_forest_schema_invalid"
    identity = content_identity(dict(portable))
    if value.get("portable_identity") != identity or forest.cid != identity:
        return False, "dcr011_forest_identity_invalid"
    return True, ""


def _verify_dcr021_graph(graph: DatasetsLogicIrEvidence) -> tuple[bool, str]:
    """Require canonical DCR-021 graph bytes and the matching graph CID."""

    from ..analysis.mcp_contract_graph import (
        MCP_CONTRACT_GRAPH_INTERFACE,
        MCP_CONTRACT_GRAPH_SCHEMA,
    )
    from .formal_verification_contracts import canonical_json_bytes, content_identity

    value = graph.payload
    if not isinstance(value, MappingABC):
        return False, "dcr021_graph_not_mapping"
    body = dict(value)
    graph_cid = body.pop("graph_cid", "")
    canonical_bytes = body.pop("canonical_bytes", "")
    if (
        body.get("schema") != MCP_CONTRACT_GRAPH_SCHEMA
        or body.get("interface") != MCP_CONTRACT_GRAPH_INTERFACE
        or body.get("authoritative") is not False
        or not isinstance(canonical_bytes, str)
        or canonical_bytes != canonical_json_bytes(body).decode("utf-8")
    ):
        return False, "dcr021_graph_schema_or_canonical_bytes_invalid"
    identity = content_identity(body)
    if graph_cid != identity or graph.cid != identity:
        return False, "dcr021_graph_identity_invalid"
    return True, ""


def _module_binding(
    *,
    module_origin: Path | str,
    capabilities: Any,
    capability_evidence: Sequence[Any] | None,
) -> tuple[dict[str, str] | None, str]:
    """Bind the exact datasets source file to typed DCR-004 receipts."""

    from ..autonomous_repair.capabilities import (
        CapabilityEvidenceReceipt,
        DeterministicRepairCapabilities,
    )

    if not isinstance(capabilities, DeterministicRepairCapabilities):
        return None, "dcr004_capability_inventory_missing"
    if not isinstance(capability_evidence, SequenceABC) or isinstance(
        capability_evidence, (str, bytes)
    ):
        return None, "dcr004_capability_evidence_missing"
    if not all(isinstance(item, CapabilityEvidenceReceipt) for item in capability_evidence):
        return None, "dcr004_capability_evidence_not_typed"
    origin = Path(module_origin)
    try:
        real_origin = origin.resolve(strict=True)
        source = real_origin.read_bytes()
    except OSError:
        return None, "datasets_logic_module_origin_unreadable"
    digest = "module:sha256:" + hashlib.sha256(source).hexdigest()
    try:
        receipt = capabilities.module(DATASETS_LOGIC_IDENTITY_MODULE)
    except StopIteration:
        return None, "datasets_logic_module_not_in_dcr004_inventory"
    if (
        not receipt.available
        or receipt.origin != str(real_origin)
        or receipt.content_digest != digest
        or not receipt.distribution_version
    ):
        return None, "datasets_logic_module_binding_mismatch"
    for kind in _REQUIRED_CAPABILITY_EVIDENCE_KINDS:
        if not any(
            item.verifies(
                evidence_id=DATASETS_LOGIC_IDENTITY_MODULE,
                evidence_kind=kind,
                subject_id=DATASETS_LOGIC_IDENTITY_MODULE,
                subject_digest=digest,
                subject_version=receipt.distribution_version,
            )
            for item in capability_evidence
        ):
            return None, "datasets_logic_capability_evidence_missing_or_stale"
    return {
        "content_digest": digest,
        "module": DATASETS_LOGIC_IDENTITY_MODULE,
        "origin": str(real_origin),
        "version": receipt.distribution_version,
    }, ""


def _invoke_datasets_logic_identity(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Use the existing real LogicIR identity facade, never a bridge result."""

    from ..integrations.ipfs_datasets_logic_provider import call_logic_ir_identity

    return call_logic_ir_identity(
        payload,
        domain="agent_supervisor.dcr030",
        schema_version=DATASETS_LOGIC_IR_INPUT_SCHEMA,
    )


def normalize_datasets_logic_ir(
    evidence: Sequence[DatasetsLogicIrEvidence],
    *,
    module_origin: Path | str,
    capabilities: Any = None,
    capability_evidence: Sequence[Any] | None = None,
) -> DatasetsLogicIrResult:
    """Normalize exact evidence with current DCR-024/DCR-004 gates.

    This operation has no model, LLM, network, proof, or mutation authority.
    A normalized identity is merely a candidate context item for a later,
    independently authorized repair workflow.
    """

    if not isinstance(evidence, SequenceABC) or isinstance(evidence, (str, bytes)):
        return _unavailable_result(DatasetsLogicIrDisposition.UNSUPPORTED, "evidence_not_typed")
    if not all(isinstance(item, DatasetsLogicIrEvidence) for item in evidence):
        return _unavailable_result(DatasetsLogicIrDisposition.UNSUPPORTED, "evidence_not_typed")
    by_kind = {item.kind: item for item in evidence}
    if len(by_kind) != len(evidence) or set(by_kind) != set(DatasetsLogicIrEvidenceKind):
        return _unavailable_result(
            DatasetsLogicIrDisposition.UNSUPPORTED,
            "evidence_must_contain_exactly_source_forest_graph_finding",
            evidence=evidence,
        )
    finding = by_kind[DatasetsLogicIrEvidenceKind.FINDING]
    graph = by_kind[DatasetsLogicIrEvidenceKind.GRAPH]
    forest = by_kind[DatasetsLogicIrEvidenceKind.FOREST]
    source = by_kind[DatasetsLogicIrEvidenceKind.SOURCE_BYTES]
    source_ok, source_reason = _verify_source_evidence(source)
    forest_ok, forest_reason = _verify_dcr011_forest(forest)
    graph_ok, graph_reason = _verify_dcr021_graph(graph)
    findings_ok, finding_reason = _verify_dcr024_findings(
        finding, graph=graph, forest=forest
    )
    binding, capability_reason = _module_binding(
        module_origin=module_origin,
        capabilities=capabilities,
        capability_evidence=capability_evidence,
    )
    if not all((source_ok, forest_ok, graph_ok, findings_ok)) or binding is None:
        reasons = [
            reason
            for reason in (
                source_reason,
                forest_reason,
                graph_reason,
                finding_reason,
                capability_reason,
            )
            if reason
        ]
        return _unavailable_result(
            DatasetsLogicIrDisposition.INTEGRATION_PENDING,
            *reasons,
            evidence=evidence,
            module_binding=binding or {},
        )
    payload = {
        "evidence": [item.to_dict() for item in sorted(evidence, key=lambda item: item.kind.value)],
        "input_cids": sorted(item.cid for item in evidence),
        "module_binding": binding,
        "schema": DATASETS_LOGIC_IR_INPUT_SCHEMA,
    }
    try:
        identity = dict(_invoke_datasets_logic_identity(payload))
    except (ImportError, ModuleNotFoundError, OSError, ValueError, TypeError) as exc:
        return _unavailable_result(
            DatasetsLogicIrDisposition.UNAVAILABLE,
            "datasets_logic_identity_unavailable",
            type(exc).__name__,
            evidence=evidence,
            module_binding=binding,
        )
    required_identity = ("cid", "digest", "profile", "logic_ir_interface")
    if any(not isinstance(identity.get(key), str) or not identity[key] for key in required_identity):
        return _unavailable_result(
            DatasetsLogicIrDisposition.UNAVAILABLE,
            "datasets_logic_identity_result_malformed",
            evidence=evidence,
            module_binding=binding,
        )
    return DatasetsLogicIrResult(
        disposition=DatasetsLogicIrDisposition.NORMALIZED,
        reason_codes=("candidate_context_only", "zero_model_calls"),
        input_cids=tuple(sorted(item.cid for item in evidence)),
        module_binding=binding,
        normalized_ir={
            "identity": identity,
            "input_schema": DATASETS_LOGIC_IR_INPUT_SCHEMA,
            "integration_status": "candidate_context_only",
        },
    )


__all__ = [
    "CONSTRAINT_ADAPTER_MODULES",
    "CONSTRAINT_ENTRYPOINTS",
    "DATASETS_LOGIC_IDENTITY_MODULE",
    "DATASETS_LOGIC_IR_FACADE_INTERFACE",
    "DATASETS_LOGIC_IR_INPUT_SCHEMA",
    "DATASETS_LOGIC_IR_RESULT_SCHEMA",
    "DEFAULT_APPLY_FAMILIES",
    "DEFAULT_DATASETS_IR_PACKAGES",
    "DEFAULT_REGISTRY_IR_FAMILIES",
    "DatasetsLogicIrDisposition",
    "DatasetsLogicIrEvidence",
    "DatasetsLogicIrEvidenceKind",
    "DatasetsLogicIrResult",
    "IR_INTEGRATION_INTERFACE",
    "IR_INTEGRATION_REPORT_SCHEMA",
    "SHARED_IR_FAMILIES",
    "STRUCTURAL_IR_FAMILIES",
    "STRUCTURAL_MODULES",
    "UI_BRIDGE_MODULES",
    "UI_DATASETS_PACKAGE",
    "IrIntegrationError",
    "IrIntegrationPolicy",
    "load_ir_policy_from_mapping",
    "normalize_datasets_logic_ir",
    "probe_constraint_adapters",
    "probe_consumer_hooks",
    "probe_datasets_ir_packages",
    "probe_ir_constraint_compiler",
    "probe_ir_integration",
    "probe_live_apply",
    "probe_registry_adapters",
    "probe_structural_ir_surfaces",
    "probe_ui_ir_surface",
]
