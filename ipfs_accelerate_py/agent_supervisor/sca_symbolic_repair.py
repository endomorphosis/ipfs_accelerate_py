"""SCA symbolic_repair orchestration via the agent supervisor.

This is the supervisor-owned entry point for residual SwissKnife contract
assurance symbolic repair.  It unifies:

* **All** datasets logic backends (IR · TDFOL · CEC · SMT · HAMMER)
* MultiProverRouter property portfolios (protocol, kernel, ATP, state, …)
* Multi-family analysis inventory (29 + Intent/Legal/Security/UI IR)
* Shared IR adapters (intent_ir · legal_ir · security_ir) + UI interface bridge
* Observation-bound claim KERNEL_VERIFIED (Lean / Coq / Isabelle)
* Repair board + RPR readiness binding

Operator scripts under ``scripts/sca_*`` are thin wrappers around this module
(or call the same pipeline stages).  LLM implement remains **proposal_only**.

Authority model (matches supervisor profile):
* kernelProof: authoritative only when exactly bound
* solverCandidate: candidate
* llmOutput: proposal_only
* claim KERNEL_VERIFIED for residual SCA: observation_bound_operator_semantics@1
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Mapping, Sequence


SCA_SYMBOLIC_REPAIR_INTERFACE: Final = "ScaSymbolicRepair@1"
SCA_SYMBOLIC_REPAIR_REPORT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/sca-symbolic-repair-stack@1"
)
CLAIM_AUTHORITY_SCOPE: Final = "observation_bound_operator_semantics@1"

# Closed set of datasets backends bound into MCP contract prover
DEFAULT_DATASETS_BACKENDS: Final[tuple[str, ...]] = (
    "ir",
    "tdfol",
    "cec",
    "smt",
    "hammer",
)

# Analysis families expected by multi-family symbolic repair (scripts)
DEFAULT_ANALYSIS_FAMILIES: Final[tuple[str, ...]] = (
    "ir",
    "software_contracts",
    "schema",
    "graph",
    "deontic",
    "cec",
    "modal",
    "tdfol",
    "event_calculus",
    "flogic",
    "smt",
    "hammer",
    "protocol",
    "proverif",
    "tamarin",
    "kernel_lean",
    "kernel_coq",
    "kernel_isabelle",
    "atp_vampire",
    "atp_e",
    "smt_cvc5",
    "authorization_datalog",
    "hyperproperty",
    "runtime_mtl",
    "state_tla",
    "state_apalache",
    "shadowprover",
    "leanstral",
    "zkp",
    # Shared-IR constraint surfaces (supervisor proof adapters)
    "intent_ir",
    "legal_ir",
    "security_ir",
    # UI/interface descriptor surface (interface_contract bridge until ui_ux_ir)
    "ui_ir",
    # Structural intermediate representations
    "ast",
    "knowledge_graph",
    "vector_index",
)

DEFAULT_KERNEL_ITPS: Final[tuple[str, ...]] = ("lean", "coq", "isabelle")


class ScaSymbolicRepairError(ValueError):
    """Malformed symbolic-repair policy or environment."""


@dataclass
class SymbolicRepairPolicy:
    """Supervisor policy for the symbolic_repair phase stack."""

    all_logic_families: bool = True
    datasets_backends: tuple[str, ...] = DEFAULT_DATASETS_BACKENDS
    analysis_families: tuple[str, ...] = DEFAULT_ANALYSIS_FAMILIES
    protocol_conformance_required: bool = True
    kernel_itps: tuple[str, ...] = DEFAULT_KERNEL_ITPS
    claim_authority_scope: str = CLAIM_AUTHORITY_SCOPE
    board_bind_required: bool = True
    max_tasks: int = 8
    max_isabelle_claims: int = 4
    max_coq_claims: int = 16
    skip_mcp_require: bool = True
    managed_provers_bin: str = ""
    repo_root: str = ""
    require_ir_integration: bool = True
    ir_integration: dict[str, Any] = field(default_factory=dict)
    require_ir_logic_apply: bool = True
    ir_logic_apply: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "SymbolicRepairPolicy":
        raw = dict(raw or {})
        backends = raw.get("datasetsBackends") or raw.get("datasets_backends")
        families = raw.get("analysisFamilies") or raw.get("analysis_families")
        itps = raw.get("kernelItps") or raw.get("kernel_itps")
        ir_raw = raw.get("irIntegration") or raw.get("ir_integration") or {}
        ir_apply_raw = raw.get("irLogicApply") or raw.get("ir_logic_apply") or {}
        return cls(
            all_logic_families=bool(
                raw.get("allLogicFamilies", raw.get("all_logic_families", True))
            ),
            datasets_backends=tuple(backends or DEFAULT_DATASETS_BACKENDS),
            analysis_families=tuple(families or DEFAULT_ANALYSIS_FAMILIES),
            protocol_conformance_required=bool(
                raw.get(
                    "protocolConformanceRequired",
                    raw.get("protocol_conformance_required", True),
                )
            ),
            kernel_itps=tuple(itps or DEFAULT_KERNEL_ITPS),
            claim_authority_scope=str(
                raw.get("claimAuthorityScope")
                or raw.get("claim_authority_scope")
                or CLAIM_AUTHORITY_SCOPE
            ),
            board_bind_required=bool(
                raw.get("boardBindRequired", raw.get("board_bind_required", True))
            ),
            max_tasks=int(raw.get("maxTasks") or raw.get("max_tasks") or 8),
            max_isabelle_claims=int(
                raw.get("maxIsabelleClaims") or raw.get("max_isabelle_claims") or 4
            ),
            max_coq_claims=int(
                raw.get("maxCoqClaims") or raw.get("max_coq_claims") or 16
            ),
            skip_mcp_require=bool(
                raw.get("skipMcpRequire", raw.get("skip_mcp_require", True))
            ),
            managed_provers_bin=str(
                raw.get("managedProversBin") or raw.get("managed_provers_bin") or ""
            ),
            repo_root=str(raw.get("repoRoot") or raw.get("repo_root") or ""),
            require_ir_integration=bool(
                raw.get(
                    "requireIrIntegration",
                    raw.get("require_ir_integration", True),
                )
            ),
            ir_integration=dict(ir_raw) if isinstance(ir_raw, Mapping) else {},
            require_ir_logic_apply=bool(
                raw.get(
                    "requireIrLogicApply",
                    raw.get("require_ir_logic_apply", True),
                )
            ),
            ir_logic_apply=dict(ir_apply_raw)
            if isinstance(ir_apply_raw, Mapping)
            else {},
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StageResult:
    name: str
    ran: bool
    exit_code: int | None = None
    error: str = ""
    report_path: str = ""
    stdout_tail: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        if self.error and not self.ran:
            return False
        if not self.ran:
            return True  # skipped
        return self.exit_code == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ran": self.ran,
            "exit_code": self.exit_code,
            "error": self.error,
            "report_path": self.report_path,
            "ok": self.ok,
            "stdout_tail": self.stdout_tail[-600:] if self.stdout_tail else "",
            "detail": self.detail,
        }


@dataclass
class SymbolicRepairStackResult:
    policy: SymbolicRepairPolicy
    stages: list[StageResult]
    snapshot_id: str = ""
    recorded_at: str = ""
    inventory: dict[str, Any] = field(default_factory=dict)
    completion_authoritative: bool = False

    @property
    def passed(self) -> bool:
        return all(s.ok for s in self.stages) and bool(self.stages)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCA_SYMBOLIC_REPAIR_REPORT_SCHEMA,
            "interface": SCA_SYMBOLIC_REPAIR_INTERFACE,
            "recorded_at": self.recorded_at
            or datetime.now(timezone.utc).isoformat(),
            "snapshot_id": self.snapshot_id,
            "completion_authoritative": self.completion_authoritative,
            "passed": self.passed,
            "policy": self.policy.to_dict(),
            "stages": [s.to_dict() for s in self.stages],
            "inventory": self.inventory,
            "notes": [
                "All datasets backends + multi-family analyzers + MultiProverRouter.",
                "Intent/Legal/Security IR adapters + UI interface bridge in inventory.",
                f"Claim KERNEL_VERIFIED authority: {self.policy.claim_authority_scope}",
                "LLM implement remains proposal_only under RPR.",
                "Board completion remains non-authoritative.",
            ],
        }


def load_policy_from_supervisor_profile(
    profile_path: str | Path | None = None,
) -> SymbolicRepairPolicy:
    """Load symbolicRepairPolicy from the SCA supervisor JSON profile."""
    if profile_path is None:
        # Walk up from this file: agent_supervisor -> ipfs_accelerate_py -> ...
        here = Path(__file__).resolve()
        candidates = [
            here.parents[4] / "config" / "swissknife_symbolic_contract_assurance_supervisor.json",
            Path.cwd()
            / "config"
            / "swissknife_symbolic_contract_assurance_supervisor.json",
        ]
        for c in candidates:
            if c.is_file():
                profile_path = c
                break
    if profile_path is None or not Path(profile_path).is_file():
        return SymbolicRepairPolicy()
    doc = json.loads(Path(profile_path).read_text(encoding="utf-8"))
    raw = dict(
        doc.get("symbolicRepairPolicy") or doc.get("symbolic_repair_policy") or {}
    )
    # Merge top-level irIntegrationPolicy when nested irIntegration is empty
    top_ir = doc.get("irIntegrationPolicy") or doc.get("ir_integration_policy") or {}
    if top_ir and not (raw.get("irIntegration") or raw.get("ir_integration")):
        raw["irIntegration"] = top_ir
    policy = SymbolicRepairPolicy.from_mapping(raw)
    root = str(doc.get("repositoryRoot") or ".")
    if not policy.repo_root:
        # Resolve relative to profile location
        base = Path(profile_path).resolve().parent.parent
        policy.repo_root = str((base / root).resolve() if root != "." else base)
    return policy


def probe_supervisor_logic_inventory(
    *,
    ir_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """In-process inventory of backends, MultiProverRouter, matrix, and IR stack."""
    inv: dict[str, Any] = {
        "datasets_backends": {},
        "routes_registered": {},
        "property_kinds": [],
        "property_policies": {},
        "analysis_families_expected": list(DEFAULT_ANALYSIS_FAMILIES),
        "kernel_itps": list(DEFAULT_KERNEL_ITPS),
        "ir_integration": {},
    }
    try:
        from .integrations.ipfs_datasets_logic_provider import (
            DatasetsLogicBackendKind,
            probe_datasets_logic_backend,
        )
        from .proof.mcp_contract_prover import (
            ContractProofRoute,
            create_mcp_contract_prover_with_datasets_logic_backends,
            datasets_logic_backends_are_registered,
        )

        prover, _reg = create_mcp_contract_prover_with_datasets_logic_backends(
            kinds=tuple(DatasetsLogicBackendKind)
        )
        for kind in DatasetsLogicBackendKind:
            try:
                p = probe_datasets_logic_backend(kind)
                inv["datasets_backends"][kind.value] = {
                    "available": bool(
                        getattr(p, "available", False)
                        or (isinstance(p, dict) and p.get("available"))
                    ),
                    "provider_id": getattr(p, "provider_id", None)
                    or (p.get("provider_id") if isinstance(p, dict) else None),
                }
            except Exception as exc:  # noqa: BLE001
                inv["datasets_backends"][kind.value] = {
                    "available": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        inv["routes_registered"] = {
            "cec": datasets_logic_backends_are_registered(
                prover, ContractProofRoute.CEC
            ),
            "tdfol": datasets_logic_backends_are_registered(
                prover, ContractProofRoute.TDFOL
            ),
            "smt": datasets_logic_backends_are_registered(
                prover, ContractProofRoute.SMT
            ),
        }
    except Exception as exc:  # noqa: BLE001
        inv["datasets_error"] = f"{type(exc).__name__}: {exc}"

    try:
        from .proof.multi_prover_router import (
            DEFAULT_PROPERTY_POLICIES,
            PropertyKind,
        )

        inv["property_kinds"] = [pk.value for pk in PropertyKind]
        inv["property_policies"] = {
            pk.value: {
                "policy_id": pol.policy_id,
                "lanes": [lane.prover_id for lane in pol.lanes],
            }
            for pk, pol in DEFAULT_PROPERTY_POLICIES.items()
        }
    except Exception as exc:  # noqa: BLE001
        inv["router_error"] = f"{type(exc).__name__}: {exc}"

    try:
        from .proof.prover_matrix_registry import probe_prover_matrix

        snap = probe_prover_matrix()
        kernels = {}
        for entry in getattr(snap, "entries", ()) or ():
            d = entry.to_dict() if hasattr(entry, "to_dict") else {}
            pid = str(d.get("prover_id") or "")
            if pid in DEFAULT_KERNEL_ITPS or pid in {
                "proverif",
                "tamarin",
                "z3",
                "vampire",
            }:
                st = d.get("states") or {}
                kernels[pid] = {
                    "highest_state": d.get("highest_state"),
                    "smoke_tested": bool(st.get("smoke_tested")),
                    "reconstruction_capable": bool(
                        st.get("reconstruction_capable")
                    ),
                }
        inv["matrix_highlights"] = kernels
        inv["matrix_entry_count"] = len(getattr(snap, "entries", ()) or ())
    except Exception as exc:  # noqa: BLE001
        inv["matrix_error"] = f"{type(exc).__name__}: {exc}"

    try:
        from .sca_ir_integration import (
            IrIntegrationPolicy,
            probe_ir_integration,
        )

        policy = (
            IrIntegrationPolicy.from_mapping(ir_policy)
            if ir_policy is not None
            else IrIntegrationPolicy()
        )
        inv["ir_integration"] = probe_ir_integration(policy)
    except Exception as exc:  # noqa: BLE001
        inv["ir_integration"] = {
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    return inv


def _repo_root(policy: SymbolicRepairPolicy) -> Path:
    if policy.repo_root:
        return Path(policy.repo_root).resolve()
    # agent_supervisor/sca_symbolic_repair.py -> parents[4] is often external parent
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "scripts" / "sca_symbolic_auto_repair_loop.py").is_file():
            return parent
        if (parent / "config" / "swissknife_symbolic_contract_assurance_supervisor.json").is_file():
            return parent
    return Path.cwd()


def _prepare_env(policy: SymbolicRepairPolicy, repo: Path) -> dict[str, str]:
    env = os.environ.copy()
    managed = Path(
        policy.managed_provers_bin
        or os.environ.get("IPFS_THEOREM_PROVERS_BIN")
        or Path.home()
        / ".local"
        / "share"
        / "ipfs_datasets_py"
        / "theorem-provers"
        / "bin"
    ).expanduser()
    parts = [str(managed)] if managed.is_dir() else []
    elan = Path.home() / ".elan" / "bin"
    if elan.is_dir():
        parts.append(str(elan))
    parts.append(env.get("PATH", ""))
    env["PATH"] = os.pathsep.join(p for p in parts if p)
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(repo / "external" / "ipfs_accelerate"),
            str(repo / "external" / "ipfs_datasets"),
            str(repo / "external" / "ipfs_kit"),
            str(repo / "Mcp-Plus-Plus"),
            str(repo / "scripts"),
            env.get("PYTHONPATH", ""),
        ]
    )
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _run_script(
    *,
    name: str,
    repo: Path,
    env: dict[str, str],
    argv: Sequence[str],
    report_rel: str,
    timeout: int,
) -> StageResult:
    cmd = [sys.executable, str(repo / "scripts" / argv[0]), *argv[1:]]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(repo),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return StageResult(
            name=name,
            ran=True,
            exit_code=proc.returncode,
            report_path=str(repo / report_rel),
            stdout_tail=(proc.stdout or "")[-800:],
            detail={"cmd": cmd},
        )
    except Exception as exc:  # noqa: BLE001
        return StageResult(
            name=name,
            ran=False,
            error=f"{type(exc).__name__}: {exc}",
            report_path=str(repo / report_rel),
        )


def run_symbolic_repair_stack(
    policy: SymbolicRepairPolicy | Mapping[str, Any] | None = None,
    *,
    stages: Sequence[str] | None = None,
) -> SymbolicRepairStackResult:
    """Run the full SCA symbolic repair stack under supervisor policy.

    Default stages:
      inventory → multi_family → full_integration → obligations →
      kernel → board_bind
    """
    if isinstance(policy, Mapping):
        policy = SymbolicRepairPolicy.from_mapping(policy)
    elif policy is None:
        policy = load_policy_from_supervisor_profile()

    repo = _repo_root(policy)
    env = _prepare_env(policy, repo)
    sca = repo / "data" / "agent_supervisor" / "swissknife_contract_assurance"
    snapshot_id = ""
    summary_path = sca / "baseline" / "runtime_components" / "summary.json"
    if summary_path.is_file():
        snapshot_id = str(
            json.loads(summary_path.read_text(encoding="utf-8")).get("snapshot_id")
            or ""
        )

    inventory = probe_supervisor_logic_inventory(
        ir_policy=dict(policy.ir_integration) if policy.ir_integration else None
    )
    inventory["analysis_families_expected"] = list(policy.analysis_families)
    inventory["all_logic_families"] = policy.all_logic_families

    want = set(
        stages
        or (
            "inventory",
            "ir_integration",
            "ir_apply",
            "multi_family",
            "full_integration",
            "obligations",
            "kernel",
            "board_bind",
            "planning",
        )
    )
    results: list[StageResult] = []

    if "inventory" in want:
        backends_ok = all(
            (inventory.get("datasets_backends") or {}).get(k, {}).get("available")
            for k in policy.datasets_backends
        )
        routes_ok = all((inventory.get("routes_registered") or {}).values())
        ir_ok = bool((inventory.get("ir_integration") or {}).get("passed", True))
        if not policy.require_ir_integration:
            ir_ok = True
        results.append(
            StageResult(
                name="inventory",
                ran=True,
                exit_code=0 if (backends_ok and routes_ok and ir_ok) else 1,
                detail={
                    "backends_ok": backends_ok,
                    "routes_ok": routes_ok,
                    "ir_integration_ok": ir_ok,
                    "property_kinds": inventory.get("property_kinds"),
                    "matrix_entry_count": inventory.get("matrix_entry_count"),
                    "ir_families": (inventory.get("ir_integration") or {}).get(
                        "families"
                    ),
                },
            )
        )

    if "ir_integration" in want:
        ir_doc = inventory.get("ir_integration") or {}
        ir_passed = bool(ir_doc.get("passed"))
        if not policy.require_ir_integration:
            ir_passed = True
        results.append(
            StageResult(
                name="ir_integration",
                ran=True,
                exit_code=0 if ir_passed else 1,
                detail={
                    "passed": ir_doc.get("passed"),
                    "gates": ir_doc.get("gates"),
                    "families": ir_doc.get("families"),
                    "ui_ir_status": (ir_doc.get("ui_ir") or {}).get("notes", [])[:2]
                    if isinstance(ir_doc.get("ui_ir"), dict)
                    else ir_doc.get("ui_ir"),
                    "analysis_families": ir_doc.get("analysis_families"),
                },
            )
        )

    if "ir_apply" in want:
        try:
            from .proof.ir_logic_application import (
                IrLogicApplyPolicy,
                apply_logic_to_surfaces,
                load_apply_policy_from_supervisor_profile,
            )

            apply_policy = load_apply_policy_from_supervisor_profile()
            if policy.ir_logic_apply:
                apply_policy = IrLogicApplyPolicy.from_mapping(
                    {**apply_policy.to_dict(), **dict(policy.ir_logic_apply)}
                )
            apply_policy.max_surfaces = max(
                apply_policy.max_surfaces, int(policy.max_tasks)
            )

            # Load residual findings (same set planning uses)
            findings: list[dict[str, Any]] = []
            for rel in (
                "baseline/runtime_components/contract_findings.json",
                "baseline/runtime_components/findings.json",
            ):
                fpath = sca / rel
                if not fpath.is_file():
                    continue
                doc = json.loads(fpath.read_text(encoding="utf-8"))
                for item in doc.get("findings") or []:
                    if not isinstance(item, dict):
                        continue
                    kind = str(item.get("kind") or item.get("reason_code") or "")
                    if kind in {
                        "observed_contract_incomplete",
                        "ambiguous_source_anchor",
                        "ambiguous_target_anchor",
                        "ambiguous_path_class",
                    }:
                        findings.append(item)
            # de-dupe by finding/contract
            by_key: dict[str, dict[str, Any]] = {}
            for item in findings:
                key = str(
                    item.get("finding_id")
                    or item.get("id")
                    or item.get("contract_id")
                    or ""
                )
                by_key[key or str(len(by_key))] = item
            selected = list(by_key.values())[: apply_policy.max_surfaces]
            # SCA residual findings are one domain of the general IR applicator
            apply_report = apply_logic_to_surfaces(
                selected,
                policy=apply_policy,
                domain="sca",
                consumer="symbolic_repair",
            )
            apply_path = (
                sca
                / "evaluation"
                / "supervisor_ir_logic_apply_report.json"
            )
            apply_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            apply_path.write_text(
                json.dumps(apply_report, indent=2, sort_keys=True, default=str)
                + "\n",
                encoding="utf-8",
            )
            apply_ok = bool(apply_report.get("passed"))
            if not policy.require_ir_logic_apply:
                apply_ok = True
            inventory["ir_logic_apply"] = {
                "passed": apply_report.get("passed"),
                "selected_count": apply_report.get("selected_count"),
                "summary": apply_report.get("summary"),
                "report": str(apply_path),
            }
            results.append(
                StageResult(
                    name="ir_apply",
                    ran=True,
                    exit_code=0 if apply_ok else 1,
                    report_path=str(apply_path),
                    detail={
                        "passed": apply_report.get("passed"),
                        "selected_count": apply_report.get("selected_count"),
                        "summary": apply_report.get("summary"),
                        "sample_family_ok": (
                            (apply_report.get("rows") or [{}])[0].get("family_ok")
                            if apply_report.get("rows")
                            else {}
                        ),
                        "logic_pipeline": [
                            "project_candidate_plan",
                            "deterministic_ir_fixture",
                            "IRRegistry.load+verify",
                            "IRAdapterRegistry.normalize",
                            "compile_intent/legal/security_constraints",
                            "evaluate_security_authorization",
                            "ui_ir interface projection",
                        ],
                    },
                )
            )
        except Exception as exc:  # noqa: BLE001
            results.append(
                StageResult(
                    name="ir_apply",
                    ran=False,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    mt = str(policy.max_tasks)
    if "multi_family" in want:
        argv = [
            "sca_multi_family_symbolic_repair.py",
            "--max-tasks",
            mt,
        ]
        if policy.all_logic_families:
            argv.append("--all-families")
        if policy.protocol_conformance_required:
            argv.append("--protocol-conformance")
        results.append(
            _run_script(
                name="multi_family",
                repo=repo,
                env=env,
                argv=argv,
                report_rel=(
                    "data/agent_supervisor/swissknife_contract_assurance/"
                    "evaluation/multi_family_symbolic_repair_report.json"
                ),
                timeout=600,
            )
        )

    if "full_integration" in want:
        argv = [
            "sca_full_prover_integration.py",
            "--max-tasks",
            mt,
            "--execute",
            "--with-obligations",
        ]
        results.append(
            _run_script(
                name="full_integration",
                repo=repo,
                env=env,
                argv=argv,
                report_rel=(
                    "data/agent_supervisor/swissknife_contract_assurance/"
                    "evaluation/full_prover_integration_report.json"
                ),
                timeout=900,
            )
        )

    if "obligations" in want:
        results.append(
            _run_script(
                name="obligations",
                repo=repo,
                env=env,
                argv=["sca_obligation_kernel_pipeline.py", "--max-tasks", mt],
                report_rel=(
                    "data/agent_supervisor/swissknife_contract_assurance/"
                    "evaluation/obligation_kernel_pipeline_report.json"
                ),
                timeout=600,
            )
        )

    if "kernel" in want:
        results.append(
            _run_script(
                name="kernel",
                repo=repo,
                env=env,
                argv=[
                    "sca_kernel_reconstruction_pipeline.py",
                    "--max-tasks",
                    str(max(policy.max_tasks, 8)),
                    "--skip-hammer",
                    "--max-isabelle-claims",
                    str(policy.max_isabelle_claims),
                    "--max-coq-claims",
                    str(policy.max_coq_claims),
                ],
                report_rel=(
                    "data/agent_supervisor/swissknife_contract_assurance/"
                    "evaluation/kernel_reconstruction_pipeline_report.json"
                ),
                timeout=1800,
            )
        )

    if "board_bind" in want and policy.board_bind_required:
        results.append(
            _run_script(
                name="board_bind",
                repo=repo,
                env=env,
                argv=["sca_bind_kernel_receipts_to_board.py"],
                report_rel=(
                    "data/agent_supervisor/swissknife_contract_assurance/"
                    "evaluation/claim_kernel_board_bind_report.json"
                ),
                timeout=120,
            )
        )

    if "planning" in want:
        # In-process symbolic planning (all families + MultiProverRouter + planner factory)
        try:
            from .sca_symbolic_planning import (
                load_planning_policy_from_supervisor_profile,
                run_symbolic_planning_stack,
                write_planning_report,
            )

            plan_policy = load_planning_policy_from_supervisor_profile()
            plan_policy.repo_root = str(repo)
            plan_policy.max_tasks = max(policy.max_tasks, plan_policy.max_tasks)
            plan_policy.all_logic_families = policy.all_logic_families
            plan_report = run_symbolic_planning_stack(plan_policy)
            # Deep bind IR apply stage into planning portfolios when both ran.
            try:
                from .planning.ir_logic_hooks import (
                    symbolic_repair_ir_portfolio_bind,
                )

                ir_report = (inventory.get("ir_logic_apply") or {})
                # Prefer full report file when stage wrote it
                ir_path = ir_report.get("report")
                ir_full = {}
                if ir_path and Path(str(ir_path)).is_file():
                    ir_full = json.loads(
                        Path(str(ir_path)).read_text(encoding="utf-8")
                    )
                plan_report = symbolic_repair_ir_portfolio_bind(
                    plan_report, ir_full or ir_report
                )
            except Exception:  # noqa: BLE001
                pass
            plan_path = write_planning_report(plan_report, repo_root=repo)
            results.append(
                StageResult(
                    name="planning",
                    ran=True,
                    exit_code=0 if plan_report.get("passed") else 1,
                    report_path=str(plan_path),
                    detail={
                        "selected_count": plan_report.get("selected_count"),
                        "gates": plan_report.get("gates"),
                        "planner_disposition": (
                            (plan_report.get("planner_stack") or {}).get(
                                "disposition"
                            )
                        ),
                        "ir_logic_deep_bound": bool(
                            plan_report.get("ir_logic_deep_bound")
                        ),
                    },
                )
            )
            inventory["symbolic_planning"] = {
                "passed": plan_report.get("passed"),
                "gates": plan_report.get("gates"),
                "report": str(plan_path),
                "ir_logic_deep_bound": bool(
                    plan_report.get("ir_logic_deep_bound")
                ),
            }
        except Exception as exc:  # noqa: BLE001
            results.append(
                StageResult(
                    name="planning",
                    ran=False,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    return SymbolicRepairStackResult(
        policy=policy,
        stages=results,
        snapshot_id=snapshot_id,
        recorded_at=datetime.now(timezone.utc).isoformat(),
        inventory=inventory,
        completion_authoritative=False,
    )


def write_stack_report(
    result: SymbolicRepairStackResult,
    path: str | Path | None = None,
) -> Path:
    repo = _repo_root(result.policy)
    if path is None:
        path = (
            repo
            / "data"
            / "agent_supervisor"
            / "swissknife_contract_assurance"
            / "evaluation"
            / "supervisor_symbolic_repair_stack_report.json"
        )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    target.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return target


__all__ = [
    "CLAIM_AUTHORITY_SCOPE",
    "DEFAULT_ANALYSIS_FAMILIES",
    "DEFAULT_DATASETS_BACKENDS",
    "DEFAULT_KERNEL_ITPS",
    "SCA_SYMBOLIC_REPAIR_INTERFACE",
    "SCA_SYMBOLIC_REPAIR_REPORT_SCHEMA",
    "ScaSymbolicRepairError",
    "StageResult",
    "SymbolicRepairPolicy",
    "SymbolicRepairStackResult",
    "load_policy_from_supervisor_profile",
    "probe_supervisor_logic_inventory",
    "run_symbolic_repair_stack",
    "write_stack_report",
]


# Re-export IR integration for operator scripts
def probe_ir_stack(
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    from .sca_ir_integration import probe_ir_integration

    return probe_ir_integration(policy)
