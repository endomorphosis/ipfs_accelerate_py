"""SCA symbolic planning via the agent supervisor.

Pairs with :mod:`sca_symbolic_repair` so residual contract work is planned with
the same full logic portfolio:

* Default planner factory (compiler / validator / replanner / adaptive)
* MultiProverRouter property-kind portfolios for every residual finding
* All analysis logic families (29 + intent/legal/security/ui IR) as planning lanes
* Datasets backends (IR/TDFOL/CEC/SMT/HAMMER)
* Shared IR constraint surfaces (Intent/Legal/Security) + UI interface bridge
* Protocol + kernel ITP inventory (ProVerif/Tamarin/Lean/Coq/Isabelle)

Does not invent KERNEL_VERIFIED claim proofs (those stay in sca_symbolic_repair).
Planning output is deterministic portfolio guidance bound to the SCA snapshot.

LLM implement remains proposal_only.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Mapping, Sequence


SCA_SYMBOLIC_PLANNING_INTERFACE: Final = "ScaSymbolicPlanning@1"
SCA_SYMBOLIC_PLANNING_REPORT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/sca-symbolic-planning-stack@1"
)

# Finding kind → MultiProverRouter PropertyKind values
FINDING_PROPERTY_KINDS: Final[dict[str, tuple[str, ...]]] = {
    "observed_contract_incomplete": (
        "temporal_deontic",
        "finite_constraint",
        "protocol",
        "typed_planning",
    ),
    "ambiguous_source_anchor": (
        "finite_constraint",
        "first_order_theorem",
        "typed_planning",
    ),
    "ambiguous_target_anchor": (
        "temporal_deontic",
        "protocol",
        "first_order_theorem",
        "typed_planning",
    ),
    "ambiguous_path_class": (
        "protocol",
        "authorization",
        "temporal_deontic",
        "hyperproperty",
    ),
}

# Finding kind → analysis families (aligned with multi-family repair)
# Shared + structural IR always bound into residual planning portfolios
_IR_LOGIC_FAMILIES: Final[tuple[str, ...]] = (
    "intent_ir",
    "legal_ir",
    "security_ir",
    "ui_ir",
    "ast",
    "knowledge_graph",
    "vector_index",
)

FINDING_ANALYSIS_FAMILIES: Final[dict[str, tuple[str, ...]]] = {
    "observed_contract_incomplete": (
        "ir",
        "software_contracts",
        "schema",
        "cec",
        "deontic",
        "smt",
        "protocol",
        "hammer",
        "kernel_lean",
        *_IR_LOGIC_FAMILIES,
    ),
    "ambiguous_source_anchor": (
        "ir",
        "software_contracts",
        "graph",
        "hammer",
        "flogic",
        "smt",
        "kernel_lean",
        *_IR_LOGIC_FAMILIES,
    ),
    "ambiguous_target_anchor": (
        "ir",
        "software_contracts",
        "cec",
        "deontic",
        "hammer",
        "protocol",
        "kernel_lean",
        *_IR_LOGIC_FAMILIES,
    ),
    "ambiguous_path_class": (
        "modal",
        "deontic",
        "cec",
        "graph",
        "ir",
        "protocol",
        "proverif",
        "tamarin",
        "authorization_datalog",
        "hyperproperty",
        *_IR_LOGIC_FAMILIES,
    ),
}

OP_PROPERTY_EXTRA: Final[list[tuple[tuple[str, ...], tuple[str, ...]]]] = [
    (
        ("dispatch", "tools_", "policy", "auth", "ucan", "session", "mcpplusplus"),
        ("protocol", "authorization", "temporal_deontic"),
    ),
    (
        ("workflow", "submit", "dag", "temporal", "schedule"),
        ("temporal_deontic", "typed_planning", "state_machine"),
    ),
    (
        ("pin", "secret", "encrypt", "attest", "zk"),
        ("protocol", "hyperproperty"),
    ),
    (
        ("runtime", "metrics", "monitor", "trace"),
        ("runtime_trace",),
    ),
]


class ScaSymbolicPlanningError(ValueError):
    """Malformed planning policy or blocked planning environment."""


@dataclass
class SymbolicPlanningPolicy:
    """Supervisor policy for symbolic planning over residual SCA work."""

    all_logic_families: bool = True
    all_property_kinds: bool = True
    require_default_planner: bool = True
    require_multi_prover_router: bool = True
    protocol_conformance_required: bool = True
    max_tasks: int = 8
    attach_claim_kernel_receipts: bool = True
    repo_root: str = ""
    require_ir_integration: bool = True
    ir_integration: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "SymbolicPlanningPolicy":
        raw = dict(raw or {})
        ir_raw = raw.get("irIntegration") or raw.get("ir_integration") or {}
        return cls(
            all_logic_families=bool(
                raw.get("allLogicFamilies", raw.get("all_logic_families", True))
            ),
            all_property_kinds=bool(
                raw.get("allPropertyKinds", raw.get("all_property_kinds", True))
            ),
            require_default_planner=bool(
                raw.get(
                    "requireDefaultPlanner",
                    raw.get("require_default_planner", True),
                )
            ),
            require_multi_prover_router=bool(
                raw.get(
                    "requireMultiProverRouter",
                    raw.get("require_multi_prover_router", True),
                )
            ),
            protocol_conformance_required=bool(
                raw.get(
                    "protocolConformanceRequired",
                    raw.get("protocol_conformance_required", True),
                )
            ),
            max_tasks=int(raw.get("maxTasks") or raw.get("max_tasks") or 8),
            attach_claim_kernel_receipts=bool(
                raw.get(
                    "attachClaimKernelReceipts",
                    raw.get("attach_claim_kernel_receipts", True),
                )
            ),
            repo_root=str(raw.get("repoRoot") or raw.get("repo_root") or ""),
            require_ir_integration=bool(
                raw.get(
                    "requireIrIntegration",
                    raw.get("require_ir_integration", True),
                )
            ),
            ir_integration=dict(ir_raw) if isinstance(ir_raw, Mapping) else {},
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_planning_policy_from_supervisor_profile(
    profile_path: str | Path | None = None,
) -> SymbolicPlanningPolicy:
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
                profile_path = c
                break
    if profile_path is None or not Path(profile_path).is_file():
        return SymbolicPlanningPolicy()
    doc = json.loads(Path(profile_path).read_text(encoding="utf-8"))
    raw = (
        doc.get("symbolicPlanningPolicy")
        or doc.get("symbolic_planning_policy")
        or {}
    )
    # Inherit allLogicFamilies from symbolicRepairPolicy when planning omits it
    repair = doc.get("symbolicRepairPolicy") or {}
    if "allLogicFamilies" not in raw and "all_logic_families" not in raw:
        raw = {
            **raw,
            "allLogicFamilies": repair.get("allLogicFamilies", True),
        }
    policy = SymbolicPlanningPolicy.from_mapping(raw)
    if not policy.repo_root:
        base = Path(profile_path).resolve().parent.parent
        root = str(doc.get("repositoryRoot") or ".")
        policy.repo_root = str((base / root).resolve() if root != "." else base)
    return policy


def _repo_root(policy: SymbolicPlanningPolicy) -> Path:
    if policy.repo_root:
        return Path(policy.repo_root).resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "scripts" / "sca_symbolic_auto_repair_loop.py").is_file():
            return parent
    return Path.cwd()


def _prepare_path() -> None:
    managed = (
        Path.home()
        / ".local"
        / "share"
        / "ipfs_datasets_py"
        / "theorem-provers"
        / "bin"
    )
    parts: list[str] = []
    if managed.is_dir():
        parts.append(str(managed))
    elan = Path.home() / ".elan" / "bin"
    if elan.is_dir():
        parts.append(str(elan))
    parts.append(os.environ.get("PATH", ""))
    os.environ["PATH"] = os.pathsep.join(p for p in parts if p)


def property_kinds_for(kind: str, op: str) -> list[str]:
    kinds = list(FINDING_PROPERTY_KINDS.get(kind, ("finite_constraint", "typed_planning")))
    op_l = op.lower()
    for tokens, extra in OP_PROPERTY_EXTRA:
        if any(t in op_l for t in tokens):
            for k in extra:
                if k not in kinds:
                    kinds.append(k)
    return kinds


def analysis_families_for(kind: str, op: str, *, all_families: bool) -> list[str]:
    if all_families:
        from .sca_symbolic_repair import DEFAULT_ANALYSIS_FAMILIES

        return list(DEFAULT_ANALYSIS_FAMILIES)
    base = list(
        FINDING_ANALYSIS_FAMILIES.get(kind, ("ir", "software_contracts", "cec"))
    )
    for tokens, extra in OP_PROPERTY_EXTRA:
        if any(t in op.lower() for t in tokens):
            # Map property extras to family names loosely
            mapping = {
                "protocol": "protocol",
                "authorization": "authorization_datalog",
                "temporal_deontic": "deontic",
                "typed_planning": "tdfol",
                "state_machine": "state_tla",
                "hyperproperty": "hyperproperty",
                "runtime_trace": "runtime_mtl",
            }
            for k in extra:
                fam = mapping.get(k)
                if fam and fam not in base:
                    base.append(fam)
    return base


def probe_planner_stack() -> dict[str, Any]:
    """Probe DefaultPlannerFactory + optional provers (PATH sealed)."""
    _prepare_path()
    out: dict[str, Any] = {"core_ready": False, "disposition": "", "provers": {}}
    try:
        from .planning.default_planner_factory import build_default_planner_handles

        handles = build_default_planner_handles()
        out["core_ready"] = bool(handles.core_ready)
        out["disposition"] = str(
            getattr(handles.disposition, "value", handles.disposition)
        )
        out["defers_capability"] = bool(handles.defers_capability)
        out["provers"] = {
            r.prover_id.value: {
                "status": r.status.value,
                "available": r.available,
                "executable": r.executable,
            }
            for r in handles.optional_prover_records
        }
        out["adaptive_planner"] = type(handles.adaptive_planner).__name__
        out["compiler"] = type(handles.compiler).__name__
        out["validator"] = type(handles.validator).__name__
        out["replanner"] = type(handles.replanner).__name__
        out["proof_carrying"] = handles.proof_carrying_handle is not None
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"

    # Extended ITP / protocol inventory (beyond factory optional set)
    for name, candidates in (
        ("isabelle", ("isabelle",)),
        ("proverif", ("proverif",)),
        ("tamarin", ("tamarin-prover", "tamarin")),
        ("vampire", ("vampire",)),
        ("e", ("eprover", "e")),
    ):
        path = None
        for c in candidates:
            path = shutil.which(c)
            if path:
                break
        out["provers"][name] = {
            "status": "available" if path else "unavailable",
            "available": bool(path),
            "executable": path or "",
        }
    return out


def probe_multi_prover_planning() -> dict[str, Any]:
    from .proof.multi_prover_router import (
        DEFAULT_PROPERTY_POLICIES,
        MultiProverRouter,
        PropertyKind,
        PropertyObligation,
    )
    from .proof.formal_verification_contracts import AssuranceLevel

    router = MultiProverRouter()
    sample_plans = {}
    for pk in PropertyKind:
        obl = PropertyObligation(
            obligation_id=f"plan-probe:{pk.value}",
            property_kind=pk,
            statement=f"planning probe for property_kind={pk.value}",
            premise_ids=(f"premise:plan-probe:{pk.value}",),
            required_assurance=AssuranceLevel.SOLVER_CHECKED,
            metadata={"source": "sca_symbolic_planning"},
        )
        plan = router.plan(obl)
        sample_plans[pk.value] = {
            "policy_id": plan.policy_id,
            "lanes": [
                {
                    "prover_id": lane.prover_id,
                    "role": lane.role.value
                    if hasattr(lane.role, "value")
                    else str(lane.role),
                    "stage": lane.stage,
                }
                for lane in plan.lanes
            ],
        }
    return {
        "property_kinds": [pk.value for pk in PropertyKind],
        "default_policies": {
            pk.value: {
                "policy_id": pol.policy_id,
                "lanes": [lane.prover_id for lane in pol.lanes],
            }
            for pk, pol in DEFAULT_PROPERTY_POLICIES.items()
        },
        "sample_plans": sample_plans,
    }


def load_residual_findings(
    repo: Path, max_tasks: int
) -> list[dict[str, Any]]:
    sca = repo / "data" / "agent_supervisor" / "swissknife_contract_assurance"
    want = {
        "observed_contract_incomplete",
        "ambiguous_source_anchor",
        "ambiguous_target_anchor",
        "ambiguous_path_class",
    }
    by_id: dict[str, dict[str, Any]] = {}
    for rel in (
        "baseline/runtime_components/contract_findings.json",
        "baseline/runtime_components/findings.json",
    ):
        path = sca / rel
        if not path.is_file():
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        for item in doc.get("findings") or []:
            if not isinstance(item, dict):
                continue
            kind = str(item.get("kind") or item.get("reason_code") or "")
            if kind not in want:
                continue
            fid = str(item.get("finding_id") or item.get("id") or "")
            key = fid or f"{kind}:{item.get('contract_id')}"
            by_id[key] = item
    return list(by_id.values())[:max_tasks]


def load_claim_kernel_index(repo: Path) -> dict[str, list[dict[str, Any]]]:
    idx_path = (
        repo
        / "data/agent_supervisor/swissknife_contract_assurance/evaluation/"
        "claim_kernel_receipts/index.json"
    )
    if not idx_path.is_file():
        return {}
    doc = json.loads(idx_path.read_text(encoding="utf-8"))
    by_contract: dict[str, list[dict[str, Any]]] = {}
    for rec in doc.get("receipts") or []:
        if not isinstance(rec, dict):
            continue
        cid = str(rec.get("contract_id") or "")
        if cid:
            by_contract.setdefault(cid, []).append(rec)
        op = cid.split(":", 1)[-1] if ":" in cid else ""
        if op:
            by_contract.setdefault(op, []).append(rec)
    return by_contract


def plan_finding_portfolio(
    item: dict[str, Any],
    *,
    snapshot_id: str,
    all_families: bool,
    all_property_kinds: bool,
    multi_prover: dict[str, Any],
    claim_index: Mapping[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Build one residual finding's symbolic planning portfolio."""
    from .proof.multi_prover_router import (
        MultiProverRouter,
        PropertyKind,
        PropertyObligation,
    )
    from .proof.formal_verification_contracts import AssuranceLevel

    contract_id = str(item.get("contract_id") or "")
    kind = str(item.get("kind") or item.get("reason_code") or "")
    finding_id = str(item.get("finding_id") or item.get("id") or "")
    op = contract_id.split(":", 1)[-1] if ":" in contract_id else contract_id

    pks = (
        list(multi_prover.get("property_kinds") or [])
        if all_property_kinds
        else property_kinds_for(kind, op)
    )
    families = analysis_families_for(kind, op, all_families=all_families)

    router = MultiProverRouter()
    portfolios: list[dict[str, Any]] = []
    for pk_name in pks:
        try:
            pk = PropertyKind(pk_name)
        except ValueError:
            continue
        obl = PropertyObligation(
            obligation_id=f"sca-plan:{finding_id or op}:{pk_name}",
            property_kind=pk,
            statement=(
                f"Plan residual SCA finding {kind} for {op} under {contract_id} "
                f"using property_kind={pk_name} (snapshot={snapshot_id or 'unknown'})."
            ),
            premise_ids=(
                f"finding:{finding_id or op}",
                f"contract:{contract_id or op}",
                f"snapshot:{snapshot_id or 'unknown'}",
            ),
            required_assurance=AssuranceLevel.SOLVER_CHECKED,
            metadata={
                "source": "sca_symbolic_planning",
                "finding_kind": kind,
                "operation": op,
                "contract_id": contract_id,
            },
        )
        plan = router.plan(obl)
        portfolios.append(
            {
                "property_kind": pk_name,
                "policy_id": plan.policy_id,
                "lanes": [
                    {
                        "prover_id": lane.prover_id,
                        "role": getattr(lane.role, "value", str(lane.role)),
                        "stage": lane.stage,
                        "authority_capability": lane.authority_capability,
                    }
                    for lane in plan.lanes
                ],
            }
        )

    # Deterministic planning steps (repair-oriented, model-free)
    steps = [
        f"Index/re-prove anchors for {op} under mcp_server preference",
        "Apply multi-family analysis portfolio (all families when policy requires)",
        "Compile McpContractObligation for claim families tied to finding kind",
        "Route MultiProverRouter portfolios for selected property kinds",
        "Discharge observation-bound claim kernel (Lean/Coq/Isabelle) when facts hold",
        "Bind claim receipts to SCA-REPAIR board + RPR readiness",
        "Re-index runtime components; re-open only if residual remains",
    ]
    if any(f in families for f in ("protocol", "proverif", "tamarin")):
        steps.insert(
            2,
            "Verify MCP mediation path (package_mcp_interop / tools/call); "
            "run ProVerif/Tamarin conformance for path_class",
        )
    if any(f.startswith("kernel_") for f in families):
        steps.insert(
            -2,
            "Prefer kernel reconstruction lanes before residual LLM packet",
        )
    ir_fams = [
        f
        for f in families
        if f
        in (
            "intent_ir",
            "legal_ir",
            "security_ir",
            "ui_ir",
            "ast",
            "knowledge_graph",
            "vector_index",
        )
    ]
    ir_apply: dict[str, Any] = {}
    if ir_fams:
        steps.insert(
            2,
            "Apply shared+structural IR logic for "
            f"{op} ({', '.join(ir_fams)}): Intent/Legal/Security/UI + "
            "AST index/query + knowledge-graph mandatory closure + "
            "deterministic vector retrieval; no execution grants",
        )
        try:
            from .proof.ir_logic_application import (
                IrLogicApplyPolicy,
                IrWorkSurface,
                apply_logic_to_ir,
            )

            ir_apply = apply_logic_to_ir(
                IrWorkSurface(
                    operation=op,
                    kind=kind,
                    contract_id=contract_id,
                    finding_id=finding_id,
                    path=str(item.get("path") or ""),
                    symbol=str(item.get("symbol") or ""),
                    domain="sca",
                    consumer="planner",
                ),
                policy=IrLogicApplyPolicy(
                    families=tuple(ir_fams),
                    evaluate_security=True,
                    include_plan_admission=False,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            ir_apply = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}

    kernel_refs = claim_index.get(contract_id) or claim_index.get(op) or []
    return {
        "contract_id": contract_id,
        "finding_id": finding_id,
        "kind": kind,
        "operation": op,
        "analysis_families": families,
        "property_kinds": pks,
        "multi_prover_portfolios": portfolios,
        "ordered_planning_steps": steps,
        "ir_surfaces": ir_fams,
        "ir_logic_apply": {
            "passed": ir_apply.get("passed"),
            "family_ok": ir_apply.get("family_ok"),
            "candidate_plan": ir_apply.get("candidate_plan"),
            "gates": ir_apply.get("gates"),
            "families": {
                name: {
                    "ok": (doc or {}).get("ok"),
                    "status": (doc or {}).get("status"),
                    "logic_applied": (doc or {}).get("logic_applied"),
                    "constraint_count": (doc or {}).get("constraint_count"),
                }
                for name, doc in (ir_apply.get("families") or {}).items()
            }
            if isinstance(ir_apply.get("families"), dict)
            else {},
            "error": ir_apply.get("error"),
        }
        if ir_fams
        else {},
        "claim_kernel_receipts": [
            {
                "receipt_id": r.get("receipt_id"),
                "target_itp": r.get("target_itp"),
                "family": r.get("family"),
            }
            for r in kernel_refs[:12]
        ],
        "allow_model": False,
        "deterministic_first": True,
    }


def run_symbolic_planning_stack(
    policy: SymbolicPlanningPolicy | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run symbolic planning inventory + residual finding portfolios."""
    if isinstance(policy, Mapping):
        policy = SymbolicPlanningPolicy.from_mapping(policy)
    elif policy is None:
        policy = load_planning_policy_from_supervisor_profile()

    repo = _repo_root(policy)
    _prepare_path()

    sca = repo / "data" / "agent_supervisor" / "swissknife_contract_assurance"
    snapshot_id = ""
    summary_path = sca / "baseline" / "runtime_components" / "summary.json"
    if summary_path.is_file():
        snapshot_id = str(
            json.loads(summary_path.read_text(encoding="utf-8")).get("snapshot_id")
            or ""
        )

    planner = probe_planner_stack()
    multi = {}
    multi_err = ""
    try:
        multi = probe_multi_prover_planning()
    except Exception as exc:  # noqa: BLE001
        multi_err = f"{type(exc).__name__}: {exc}"

    from .sca_symbolic_repair import probe_supervisor_logic_inventory

    logic_inv = probe_supervisor_logic_inventory(
        ir_policy=dict(policy.ir_integration) if policy.ir_integration else None
    )
    ir_doc = logic_inv.get("ir_integration") or {}
    ir_ok = bool(ir_doc.get("passed")) if policy.require_ir_integration else True

    findings = load_residual_findings(repo, policy.max_tasks)
    claim_index = (
        load_claim_kernel_index(repo) if policy.attach_claim_kernel_receipts else {}
    )

    portfolios = []
    for item in findings:
        try:
            portfolios.append(
                plan_finding_portfolio(
                    item,
                    snapshot_id=snapshot_id,
                    all_families=policy.all_logic_families,
                    all_property_kinds=policy.all_property_kinds,
                    multi_prover=multi,
                    claim_index=claim_index,
                )
            )
        except Exception as exc:  # noqa: BLE001
            portfolios.append(
                {
                    "contract_id": item.get("contract_id"),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    # Smoke AdaptivePlanner.plan_symbolically availability (no full graph)
    adaptive_ok = False
    adaptive_detail = ""
    try:
        from .planning.adaptive_planner import AdaptivePlanner

        adaptive_ok = callable(getattr(AdaptivePlanner, "plan_symbolically", None))
        adaptive_detail = "plan_symbolically_available"
    except Exception as exc:  # noqa: BLE001
        adaptive_detail = f"{type(exc).__name__}: {exc}"

    planner_ok = bool(planner.get("core_ready")) if policy.require_default_planner else True
    router_ok = bool(multi.get("property_kinds")) if policy.require_multi_prover_router else True
    backends_ok = all(
        (logic_inv.get("datasets_backends") or {}).get(k, {}).get("available")
        for k in ("ir", "tdfol", "cec", "smt", "hammer")
    )
    portfolio_ok = bool(portfolios) and all(
        "error" not in p for p in portfolios
    )
    families_ok = (
        not policy.all_logic_families
        or all(
            len(p.get("analysis_families") or []) >= 20
            for p in portfolios
            if "error" not in p
        )
    )

    report = {
        "schema": SCA_SYMBOLIC_PLANNING_REPORT_SCHEMA,
        "interface": SCA_SYMBOLIC_PLANNING_INTERFACE,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "snapshot_id": snapshot_id,
        "completion_authoritative": False,
        "policy": policy.to_dict(),
        "planner_stack": planner,
        "multi_prover_planning": multi if multi else {"error": multi_err},
        "logic_inventory": {
            "datasets_backends": logic_inv.get("datasets_backends"),
            "routes_registered": logic_inv.get("routes_registered"),
            "property_kinds": logic_inv.get("property_kinds"),
            "matrix_entry_count": logic_inv.get("matrix_entry_count"),
            "ir_integration": {
                "passed": ir_doc.get("passed"),
                "gates": ir_doc.get("gates"),
                "families": ir_doc.get("families"),
            },
        },
        "adaptive_planner": {
            "plan_symbolically_available": adaptive_ok,
            "detail": adaptive_detail,
        },
        "selected_count": len(findings),
        "portfolios": portfolios,
        "gates": {
            "default_planner_ready": planner_ok,
            "multi_prover_router": router_ok,
            "datasets_backends": backends_ok,
            "portfolios_ok": portfolio_ok,
            "all_families_on_portfolios": families_ok,
            "adaptive_plan_symbolically": adaptive_ok,
            "ir_integration": ir_ok,
        },
        "notes": [
            "Symbolic planning attaches MultiProverRouter + all analysis families "
            "to residual SCA findings before residual LLM packets.",
            "DefaultPlannerFactory binds FormalPlanCompiler/Validator/Replanner/AdaptivePlanner.",
            "Intent/Legal/Security/UI + AST/KG/vector IR applied in family portfolios.",
            "Planner portfolios carry ir_logic_apply receipts (non-authoritative).",
            "Claim kernel receipts are linked when present (from sca_symbolic_repair).",
            "LLM remains proposal_only; planning is deterministic-first.",
        ],
        "passed": (
            planner_ok
            and router_ok
            and backends_ok
            and portfolio_ok
            and families_ok
            and adaptive_ok
            and ir_ok
            and not multi_err
        ),
    }
    return report


def write_planning_report(
    report: Mapping[str, Any],
    path: str | Path | None = None,
    *,
    repo_root: str | Path | None = None,
) -> Path:
    if path is None:
        roots: list[Path] = []
        if repo_root:
            roots.append(Path(repo_root))
        pol = report.get("policy") if isinstance(report.get("policy"), dict) else {}
        if pol.get("repo_root"):
            roots.append(Path(str(pol["repo_root"])))
        roots.extend(_possible_repos())
        roots.append(Path.cwd())
        path = None
        for parent in roots:
            if (parent / "config" / "swissknife_symbolic_contract_assurance_supervisor.json").is_file() or (
                parent / "data" / "agent_supervisor"
            ).is_dir():
                path = (
                    parent
                    / "data/agent_supervisor/swissknife_contract_assurance/evaluation/"
                    "supervisor_symbolic_planning_stack_report.json"
                )
                break
        if path is None:
            path = Path("supervisor_symbolic_planning_stack_report.json")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    target.write_text(
        json.dumps(dict(report), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return target


def _possible_repos() -> list[Path]:
    here = Path(__file__).resolve()
    out = []
    for p in here.parents:
        if (
            p / "config" / "swissknife_symbolic_contract_assurance_supervisor.json"
        ).is_file():
            out.append(p)
    return out


__all__ = [
    "FINDING_ANALYSIS_FAMILIES",
    "FINDING_PROPERTY_KINDS",
    "SCA_SYMBOLIC_PLANNING_INTERFACE",
    "SCA_SYMBOLIC_PLANNING_REPORT_SCHEMA",
    "ScaSymbolicPlanningError",
    "SymbolicPlanningPolicy",
    "analysis_families_for",
    "load_planning_policy_from_supervisor_profile",
    "plan_finding_portfolio",
    "probe_multi_prover_planning",
    "probe_planner_stack",
    "property_kinds_for",
    "run_symbolic_planning_stack",
    "write_planning_report",
]
