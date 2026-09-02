"""Fail-closed PCPR-030 Accelerate legacy mock-coordinator quarantine.

PCPR-030 moves MockWorker and the fallback-without-backend workflow
coordinator behind an explicit compatibility/simulation namespace.
Ordinary runtime cannot instantiate them. This module is not release
authority: it does not write DuckDB or Quack state, does not qualify
CPU/CUDA, and never emits a closed PCPR release outcome.

Live claims require live evidence. Simulated coordinator results stay
``Simulated``. Missing live workers stay typed unavailable.
"""

from __future__ import annotations

import ast
import hashlib
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from ..proof.formal_verification_contracts import content_identity
from .canonical_supervisor_contract_freeze import CLOSED_RELEASE_OUTCOMES
from .direct_objective_event_driven_qualification import (
    CURRENT_HEAD_UNAVAILABLE_VERDICT_CID,
    qualify_current_head_without_live_campaign,
)
from .legacy_bypass_and_false_authority_inventory import (
    CURRENT_HEAD_NON_PROMOTION_VERDICT_CID as PCPR_003_VERDICT_CID,
)
from .source_seal_and_supervisor_baseline import (
    SEALED_GIT_BINARY,
    SEALED_PATH,
    SEALED_PYTHON,
)


QUARANTINE_INTERFACE: Final = "LegacyMockCoordinatorQuarantine@1"
QUARANTINE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/legacy-mock-coordinator-quarantine@1"
)
QUARANTINE_VERDICT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "legacy-mock-coordinator-quarantine-verdict@1"
)

PCPR_030_TASK_ID: Final = "PCPR-030"
PCPR_030_GOAL_ID: Final = "PCPR-G410"
PCPR_003_TASK_ID: Final = "PCPR-003"
PCPR_001_TASK_ID: Final = "PCPR-001"
PCPR_PROGRAM_ID: Final = "proof-carrying-platform-qualification-and-release-v1"
PCPR_BOARD_NAMESPACE: Final = "proof-carrying-platform-qualification-and-release-v1"

EVIDENCE_KINDS: Final[frozenset[str]] = frozenset(
    {
        "measured",
        "measured_live",
        "measured_hermetic",
        "estimated",
        "simulated",
        "unavailable",
    }
)
PROMOTION_STATUSES: Final[frozenset[str]] = frozenset(
    {
        "supervisor_promoted",
        "supervisor_non_promoted",
        "rnd_non_promoted",
        "typed_unavailable",
        "typed_blocked",
    }
)

CURRENT_HEAD_OUTER_COMMIT: Final = "cabd50a2b61217ee60003e27d7b54cee1700a3e6"
CURRENT_HEAD_OUTER_TREE: Final = "155a0d0b836334772190001c67dee27349edc6ab"
CURRENT_HEAD_OUTER_SUBJECT: Final = (
    "fix(pcpr): reseal accelerate planning pins after reboot"
)
CURRENT_HEAD_OUTER_BRANCH: Final = (
    "implementation/pcpr-030-f3d990db7a3d-attempt-1-1788310349"
)
CURRENT_HEAD_ORIGIN_MAIN: Final = "bb8869ed72eb7002434345d9969efee729c4f7f6"
CURRENT_HEAD_FIRST_PARENT: Final = "ccae995b629574b0ede086dcf98b65b832dbdbbb"
CURRENT_HEAD_ACCELERATOR_COMMIT: Final = (
    "32764c3c0f78c03f2d0bb2cdee83315a99afcb0d"
)
CURRENT_HEAD_ACCELERATOR_TREE: Final = "db565768802ba3d13e8c41e966071bcf378a7c06"
CURRENT_HEAD_ACCELERATOR_ORIGIN_MAIN: Final = (
    "f8c2f633fa6a781b822176fd63e1a229f96b581c"
)
CURRENT_HEAD_DATASETS_COMMIT: Final = "f49afc579c22856849ca9f739435e5820003384f"
CURRENT_HEAD_DATASETS_TREE: Final = "47118a8e6d1b6b4e7ae04f9a2efda33aadebca1b"
CURRENT_HEAD_KIT_COMMIT: Final = "b6c65ba732733d7e33852713ba18aa3b12235668"
CURRENT_HEAD_KIT_TREE: Final = "14da7d92e130b7ba3523d0d6741a3ef7ef1e1bc2"

HERMETIC_CANDIDATE_SUITES: Final[tuple[str, ...]] = (
    "test/api/test_agent_supervisor_legacy_mock_coordinator_quarantine.py",
)

SIMULATION_MODULE_RELPATH: Final = (
    "ipfs_accelerate_py/compatibility/simulation/legacy_mock_coordinator.py"
)
ORDINARY_RUNTIME_RELPATH: Final = "ipfs_accelerate_py/ipfs_accelerate.py"
WORKFLOW_RELPATH: Final = "ipfs_accelerate_py/datasets_integration/workflow.py"
MCP_TOOL_RELPATH: Final = (
    "ipfs_accelerate_py/mcp_server/tools/workflow/native_workflow_tools.py"
)
LEGACY_MODULE_RELPATH: Final = "ipfs_accelerate_py/ipfs_accelerate_py_legacy.py"

COMMIT_RE: Final = __import__("re").compile(r"^[0-9a-f]{40}$")


class LegacyMockCoordinatorQuarantineError(ValueError):
    """Malformed quarantine evidence or a forbidden authority claim."""


@dataclass(frozen=True)
class QuarantineProbe:
    """One measured or typed-unavailable quarantine observation."""

    probe_id: str
    present: bool | None
    evidence_kind: str
    live: bool
    simulated_represented_as_live: bool
    reason: str
    details: Mapping[str, Any] = MappingProxyType({})

    def to_mapping(self) -> dict[str, Any]:
        return {
            "probe_id": self.probe_id,
            "present": self.present,
            "evidence_kind": self.evidence_kind,
            "live": self.live,
            "simulated_represented_as_live": self.simulated_represented_as_live,
            "reason": self.reason,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class QuarantineVerdict:
    """Fail-closed PCPR-030 quarantine decision. Never a closed release."""

    schema: str
    interface: str
    promotion_status: str
    supervisor_disposition: str
    closed_release_outcome: None
    release_claim: bool
    completion_authoritative: bool
    contracts_frozen: bool
    duckdb_or_quack_state_written: bool
    ordinary_runtime_instantiates_mock_worker: bool
    mock_worker_quarantined: bool
    workflow_coordinator_requires_explicit_simulation: bool
    simulated_results_represented_as_live: bool
    live_cuda_qualified: bool
    live_cuda_evidence_kind: str
    this_task_created_competing_authority: bool
    probes: tuple[QuarantineProbe, ...]
    blockers: tuple[str, ...]
    verdict_cid: str

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": self.interface,
            "promotion_status": self.promotion_status,
            "supervisor_disposition": self.supervisor_disposition,
            "closed_release_outcome": self.closed_release_outcome,
            "release_claim": self.release_claim,
            "completion_authoritative": self.completion_authoritative,
            "contracts_frozen": self.contracts_frozen,
            "duckdb_or_quack_state_written": self.duckdb_or_quack_state_written,
            "ordinary_runtime_instantiates_mock_worker": (
                self.ordinary_runtime_instantiates_mock_worker
            ),
            "mock_worker_quarantined": self.mock_worker_quarantined,
            "workflow_coordinator_requires_explicit_simulation": (
                self.workflow_coordinator_requires_explicit_simulation
            ),
            "simulated_results_represented_as_live": (
                self.simulated_results_represented_as_live
            ),
            "live_cuda_qualified": self.live_cuda_qualified,
            "live_cuda_evidence_kind": self.live_cuda_evidence_kind,
            "this_task_created_competing_authority": (
                self.this_task_created_competing_authority
            ),
            "probes": [item.to_mapping() for item in self.probes],
            "blockers": list(self.blockers),
            "verdict_cid": self.verdict_cid,
        }


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LegacyMockCoordinatorQuarantineError(f"{name} must be a non-empty string")
    return value.strip()


def _kind(value: Any, name: str) -> str:
    kind = _text(value, name)
    if kind not in EVIDENCE_KINDS:
        raise LegacyMockCoordinatorQuarantineError(f"{name} is not an admitted evidence kind")
    return kind


def _git_object_id(value: Any, name: str) -> str:
    text = _text(value, name)
    if COMMIT_RE.fullmatch(text) is None:
        raise LegacyMockCoordinatorQuarantineError(
            f"{name} must be a 40-character lowercase git object id"
        )
    return text


def _reject_closed_release_value(value: Any, name: str) -> None:
    if isinstance(value, str) and value in CLOSED_RELEASE_OUTCOMES:
        raise LegacyMockCoordinatorQuarantineError(
            f"{name} must not be a closed PCPR release outcome"
        )


def _require_ancestor(flag: bool, name: str) -> None:
    if flag is not True:
        raise LegacyMockCoordinatorQuarantineError(f"{name} must be true")


def discover_accelerate_root(start: Path | None = None) -> Path | None:
    here = Path(start or __file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "ipfs_accelerate_py").is_dir() and (
            candidate / "pyproject.toml"
        ).is_file():
            return candidate
    return None


def discover_portfolio_root(start: Path | None = None) -> Path | None:
    here = Path(start or __file__).resolve()
    for candidate in (here, *here.parents):
        marker = (
            candidate
            / "artifacts"
            / "proof_carrying_platform_qualification_and_release"
        )
        accel = candidate / "external" / "ipfs_accelerate"
        if marker.is_dir() and accel.is_dir():
            return candidate
    return None


def _read_source(root: Path, relpath: str) -> str | None:
    path = root / relpath
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _constructor_instantiates_mock_worker(source: str) -> bool:
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != "ipfs_accelerate_py":
            continue
        for item in node.body:
            if not isinstance(item, ast.FunctionDef) or item.name != "__init__":
                continue
            for child in ast.walk(item):
                if not isinstance(child, ast.Call):
                    continue
                func = child.func
                if isinstance(func, ast.Name) and func.id == "MockWorker":
                    return True
                if isinstance(func, ast.Name) and func.id == "type" and child.args:
                    first = child.args[0]
                    if isinstance(first, ast.Constant) and first.value == "MockWorker":
                        return True
            return False
    return False


def _constructor_loads_ordinary_worker(source: str) -> bool:
    return "load_ordinary_runtime_worker" in source


def _workflow_admits_via_quarantine(source: str) -> bool:
    return "admit_legacy_workflow_coordinator" in source and "explicit_simulation" in source


def current_head_quarantine_probes(
    *,
    accelerate_root: Path | None = None,
) -> tuple[QuarantineProbe, ...]:
    """Measured current-tree probes. Missing files stay typed unavailable."""

    root = accelerate_root or discover_accelerate_root()
    probes: list[QuarantineProbe] = []
    if root is None or not root.is_dir():
        return (
            QuarantineProbe(
                probe_id="accelerate_source_tree",
                present=None,
                evidence_kind="unavailable",
                live=False,
                simulated_represented_as_live=False,
                reason="Accelerate source tree is not present and is not recorded as empty.",
            ),
        )

    ordinary = _read_source(root, ORDINARY_RUNTIME_RELPATH)
    simulation = _read_source(root, SIMULATION_MODULE_RELPATH)
    workflow = _read_source(root, WORKFLOW_RELPATH)
    mcp = _read_source(root, MCP_TOOL_RELPATH)
    legacy = _read_source(root, LEGACY_MODULE_RELPATH)

    if ordinary is None:
        probes.append(
            QuarantineProbe(
                probe_id="ordinary_runtime_constructor",
                present=None,
                evidence_kind="unavailable",
                live=False,
                simulated_represented_as_live=False,
                reason="ipfs_accelerate.py is unreadable; absence is not recorded as passing.",
            )
        )
    else:
        instantiates = _constructor_instantiates_mock_worker(ordinary)
        loads = _constructor_loads_ordinary_worker(ordinary)
        probes.append(
            QuarantineProbe(
                probe_id="ordinary_runtime_constructor",
                present=instantiates,
                evidence_kind="measured",
                live=False,
                simulated_represented_as_live=False,
                reason=(
                    "Ordinary ipfs_accelerate_py.__init__ still constructs MockWorker."
                    if instantiates
                    else "Ordinary constructor does not instantiate MockWorker."
                ),
                details=MappingProxyType(
                    {
                        "loads_ordinary_runtime_worker": loads,
                        "relpath": ORDINARY_RUNTIME_RELPATH,
                    }
                ),
            )
        )

    if simulation is None:
        probes.append(
            QuarantineProbe(
                probe_id="simulation_namespace_mock_worker",
                present=None,
                evidence_kind="unavailable",
                live=False,
                simulated_represented_as_live=False,
                reason="Simulation namespace module is unreadable.",
            )
        )
    else:
        present = "class MockWorker" in simulation
        probes.append(
            QuarantineProbe(
                probe_id="simulation_namespace_mock_worker",
                present=present,
                evidence_kind="measured",
                live=False,
                simulated_represented_as_live=False,
                reason=(
                    "MockWorker is defined in the explicit simulation namespace."
                    if present
                    else "MockWorker is missing from the simulation namespace."
                ),
                details=MappingProxyType({"relpath": SIMULATION_MODULE_RELPATH}),
            )
        )

    if workflow is None:
        probes.append(
            QuarantineProbe(
                probe_id="workflow_coordinator_gate",
                present=None,
                evidence_kind="unavailable",
                live=False,
                simulated_represented_as_live=False,
                reason="WorkflowCoordinator source is unreadable.",
            )
        )
    else:
        gated = _workflow_admits_via_quarantine(workflow)
        probes.append(
            QuarantineProbe(
                probe_id="workflow_coordinator_gate",
                present=gated,
                evidence_kind="measured",
                live=False,
                simulated_represented_as_live=False,
                reason=(
                    "WorkflowCoordinator requires explicit simulation admission."
                    if gated
                    else "WorkflowCoordinator is still instantiable without explicit simulation."
                ),
                details=MappingProxyType({"relpath": WORKFLOW_RELPATH}),
            )
        )

    if mcp is None:
        probes.append(
            QuarantineProbe(
                probe_id="mcp_ordinary_path",
                present=None,
                evidence_kind="unavailable",
                live=False,
                simulated_represented_as_live=False,
                reason="MCP workflow tool source is unreadable.",
            )
        )
    else:
        typed = (
            "LegacyMockCoordinatorError" in mcp
            and "quarantined_coordinator_unavailable_result" in mcp
        )
        probes.append(
            QuarantineProbe(
                probe_id="mcp_ordinary_path",
                present=typed,
                evidence_kind="measured",
                live=False,
                simulated_represented_as_live=False,
                reason=(
                    "MCP ordinary path returns typed unavailable for the quarantined coordinator."
                    if typed
                    else "MCP ordinary path does not catch the quarantine error as typed unavailable."
                ),
                details=MappingProxyType({"relpath": MCP_TOOL_RELPATH}),
            )
        )

    if legacy is None:
        probes.append(
            QuarantineProbe(
                probe_id="legacy_compatibility_module",
                present=None,
                evidence_kind="unavailable",
                live=False,
                simulated_represented_as_live=False,
                reason="ipfs_accelerate_py_legacy.py is unreadable.",
            )
        )
    else:
        marked = "PCPR_030_COMPATIBILITY_SURFACE" in legacy and "ORDINARY_RUNTIME = False" in legacy
        probes.append(
            QuarantineProbe(
                probe_id="legacy_compatibility_module",
                present=marked,
                evidence_kind="measured",
                live=False,
                simulated_represented_as_live=False,
                reason=(
                    "Legacy compatibility module is marked not ordinary runtime."
                    if marked
                    else "Legacy compatibility module is not explicitly marked."
                ),
                details=MappingProxyType({"relpath": LEGACY_MODULE_RELPATH}),
            )
        )

    runtime_probes = _runtime_quarantine_probes()
    probes.extend(runtime_probes)
    return tuple(probes)


def _runtime_quarantine_probes() -> tuple[QuarantineProbe, ...]:
    from ipfs_accelerate_py.compatibility.simulation.legacy_mock_coordinator import (
        LegacyMockCoordinatorError,
        MockWorker,
        UnavailableWorker,
        instantiate_mock_worker,
        load_ordinary_runtime_worker,
        quarantined_coordinator_unavailable_result,
    )
    from ipfs_accelerate_py.datasets_integration.workflow import WorkflowCoordinator
    from ipfs_accelerate_py.mcp_server.tools.workflow.native_workflow_tools import (
        workflow_coordinator_submit_task,
    )

    ordinary = load_ordinary_runtime_worker()
    ordinary_ok = isinstance(ordinary, UnavailableWorker) and ordinary.live is False
    hardware = ordinary.test_hardware()
    cuda_live = hardware.get("cuda") is True
    probes = [
        QuarantineProbe(
            probe_id="ordinary_runtime_worker",
            present=ordinary_ok,
            evidence_kind="measured",
            live=False,
            simulated_represented_as_live=False,
            reason=(
                "Ordinary runtime worker is typed unavailable and is not MockWorker."
                if ordinary_ok
                else "Ordinary runtime worker is not the typed unavailable stand-in."
            ),
            details=MappingProxyType(
                {
                    "type_name": type(ordinary).__name__,
                    "origin": getattr(ordinary, "origin", None),
                    "cuda": hardware.get("cuda"),
                    "outcome": hardware.get("outcome"),
                }
            ),
        )
    ]

    mock_blocked = False
    mock_code = ""
    try:
        instantiate_mock_worker()
    except LegacyMockCoordinatorError as exc:
        mock_blocked = True
        mock_code = exc.code
    probes.append(
        QuarantineProbe(
            probe_id="mock_worker_ordinary_instantiation",
            present=mock_blocked,
            evidence_kind="measured",
            live=False,
            simulated_represented_as_live=False,
            reason=(
                "MockWorker ordinary instantiation is refused."
                if mock_blocked
                else "MockWorker can still be constructed without explicit simulation."
            ),
            details=MappingProxyType({"code": mock_code}),
        )
    )

    simulated = instantiate_mock_worker(explicit_simulation=True)
    simulated_hw = simulated.test_hardware()
    simulated_as_live = bool(simulated.live) or simulated_hw.get("cuda") is True
    probes.append(
        QuarantineProbe(
            probe_id="mock_worker_explicit_simulation",
            present=isinstance(simulated, MockWorker),
            evidence_kind="measured",
            live=False,
            simulated_represented_as_live=simulated_as_live,
            reason=(
                "Explicit MockWorker is simulated and does not report live CUDA."
                if not simulated_as_live
                else "Explicit MockWorker reported a live CUDA claim."
            ),
            details=MappingProxyType(
                {
                    "origin": simulated.origin,
                    "outcome": simulated_hw.get("outcome"),
                    "cuda": simulated_hw.get("cuda"),
                }
            ),
        )
    )

    workflow_blocked = False
    try:
        WorkflowCoordinator()
    except LegacyMockCoordinatorError:
        workflow_blocked = True
    probes.append(
        QuarantineProbe(
            probe_id="workflow_coordinator_ordinary_instantiation",
            present=workflow_blocked,
            evidence_kind="measured",
            live=False,
            simulated_represented_as_live=False,
            reason=(
                "Ordinary WorkflowCoordinator instantiation is refused."
                if workflow_blocked
                else "WorkflowCoordinator can still be constructed without explicit simulation."
            ),
        )
    )

    with tempfile.TemporaryDirectory(prefix="pcpr-030-") as tmp:
        coordinator = WorkflowCoordinator(
            {"cache_dir": tmp, "explicit_simulation": True}
        )
        simulated_status = {
            "origin": coordinator.origin,
            "live": coordinator.live,
            "outcome": "Simulated",
            "p2p_enabled": coordinator.enabled,
        }
        simulated_live_claim = bool(coordinator.live)
        simulated_present = (
            coordinator.live is False and coordinator.origin == "simulated"
        )
    probes.append(
        QuarantineProbe(
            probe_id="workflow_coordinator_explicit_simulation",
            present=simulated_present,
            evidence_kind="measured",
            live=False,
            simulated_represented_as_live=simulated_live_claim,
            reason=(
                "Explicit WorkflowCoordinator is labeled simulated and not live."
                if simulated_present
                else "Explicit WorkflowCoordinator claimed live."
            ),
            details=MappingProxyType(simulated_status),
        )
    )

    mcp_result = workflow_coordinator_submit_task(
        task_id="pcpr-030-ordinary",
        task_type="quarantine-probe",
        data={"origin": "ordinary"},
    )
    mcp_unavailable = (
        mcp_result.get("status") == "unavailable"
        and mcp_result.get("success") is False
        and mcp_result.get("outcome") == "Unavailable"
        and mcp_result.get("live") is False
    )
    probes.append(
        QuarantineProbe(
            probe_id="mcp_submit_ordinary_path",
            present=mcp_unavailable,
            evidence_kind="measured",
            live=False,
            simulated_represented_as_live=mcp_result.get("success") is True,
            reason=(
                "MCP ordinary submit_task returns typed unavailable."
                if mcp_unavailable
                else "MCP ordinary submit_task did not return typed unavailable."
            ),
            details=MappingProxyType(
                {
                    "status": mcp_result.get("status"),
                    "outcome": mcp_result.get("outcome"),
                    "code": mcp_result.get("code"),
                }
            ),
        )
    )

    envelope = quarantined_coordinator_unavailable_result(task_id="pcpr-030")
    probes.append(
        QuarantineProbe(
            probe_id="typed_unavailable_envelope",
            present=envelope.get("outcome") == "Unavailable" and envelope.get("live") is False,
            evidence_kind="measured",
            live=False,
            simulated_represented_as_live=False,
            reason="Typed unavailable envelope does not claim live success.",
        )
    )

    probes.append(
        QuarantineProbe(
            probe_id="live_cuda_qualification",
            present=None,
            evidence_kind="unavailable",
            live=False,
            simulated_represented_as_live=False,
            reason=(
                "This task does not qualify live CUDA. Missing GPU evidence stays "
                "typed unavailable and is not recorded as False or passing. "
                f"ordinary_worker_cuda={hardware.get('cuda')!r} cuda_live={cuda_live}."
            ),
        )
    )
    return tuple(probes)


def qualify_legacy_mock_coordinator_quarantine(
    *,
    probes: Sequence[QuarantineProbe],
    duckdb_or_quack_state_written: bool = False,
) -> QuarantineVerdict:
    """Evaluate PCPR-030. Quarantine is fail-closed; promotion is never a release."""

    if duckdb_or_quack_state_written:
        raise LegacyMockCoordinatorQuarantineError(
            "legacy mock-coordinator quarantine must not write DuckDB or Quack state"
        )
    if not probes:
        raise LegacyMockCoordinatorQuarantineError("quarantine probes are required")

    normalized: list[QuarantineProbe] = []
    blockers: list[str] = []
    simulated_as_live = False
    for probe in probes:
        probe_id = _text(probe.probe_id, "probe_id")
        kind = _kind(probe.evidence_kind, f"{probe_id}.evidence_kind")
        if kind == "estimated":
            raise LegacyMockCoordinatorQuarantineError(
                f"{probe_id}: estimated values cannot mint quarantine evidence"
            )
        if kind == "simulated" and probe.present is True:
            raise LegacyMockCoordinatorQuarantineError(
                "simulated observations cannot be represented as live presence"
            )
        if kind == "measured_live":
            raise LegacyMockCoordinatorQuarantineError(
                f"{probe_id}: this quarantine evaluator cannot carry measured_live evidence"
            )
        if probe.simulated_represented_as_live:
            simulated_as_live = True
            blockers.append(f"{probe_id}:simulated_represented_as_live")
        if probe.live:
            blockers.append(f"{probe_id}:live_claim")
        normalized.append(
            QuarantineProbe(
                probe_id=probe_id,
                present=probe.present,
                evidence_kind=kind,
                live=bool(probe.live),
                simulated_represented_as_live=bool(probe.simulated_represented_as_live),
                reason=str(probe.reason or ""),
                details=MappingProxyType(dict(probe.details)),
            )
        )

    by_id = {item.probe_id: item for item in normalized}
    instantiates = by_id.get("ordinary_runtime_constructor")
    ordinary_instantiates = bool(instantiates and instantiates.present is True)
    if ordinary_instantiates:
        blockers.append("ordinary_runtime_instantiates_mock_worker")

    mock_ns = by_id.get("simulation_namespace_mock_worker")
    mock_quarantined = bool(mock_ns and mock_ns.present is True)
    if mock_ns is None or mock_ns.present is not True:
        blockers.append("mock_worker_not_in_simulation_namespace")

    gate = by_id.get("workflow_coordinator_gate")
    workflow_gated = bool(gate and gate.present is True)
    if not workflow_gated:
        blockers.append("workflow_coordinator_not_gated")

    ordinary_worker = by_id.get("ordinary_runtime_worker")
    if ordinary_worker is None or ordinary_worker.present is not True:
        blockers.append("ordinary_runtime_worker_not_unavailable")

    mock_blocked = by_id.get("mock_worker_ordinary_instantiation")
    if mock_blocked is None or mock_blocked.present is not True:
        blockers.append("mock_worker_ordinary_instantiation_allowed")

    workflow_blocked = by_id.get("workflow_coordinator_ordinary_instantiation")
    if workflow_blocked is None or workflow_blocked.present is not True:
        blockers.append("workflow_coordinator_ordinary_instantiation_allowed")

    mcp = by_id.get("mcp_submit_ordinary_path")
    if mcp is None or mcp.present is not True:
        blockers.append("mcp_ordinary_path_not_typed_unavailable")

    unavailable_count = sum(1 for item in normalized if item.evidence_kind == "unavailable")
    if blockers:
        promotion_status = "typed_blocked"
    elif unavailable_count == len(normalized):
        promotion_status = "typed_unavailable"
    else:
        promotion_status = "rnd_non_promoted"

    if promotion_status not in PROMOTION_STATUSES:
        raise LegacyMockCoordinatorQuarantineError("internal promotion status is not admitted")
    if promotion_status in CLOSED_RELEASE_OUTCOMES:
        raise LegacyMockCoordinatorQuarantineError(
            "legacy mock-coordinator quarantine must not mint a closed release outcome"
        )

    payload = {
        "schema": QUARANTINE_VERDICT_SCHEMA,
        "interface": QUARANTINE_INTERFACE,
        "task_id": PCPR_030_TASK_ID,
        "goal_id": PCPR_030_GOAL_ID,
        "promotion_status": promotion_status,
        "supervisor_disposition": "supervisor_non_promoted",
        "closed_release_outcome": None,
        "release_claim": False,
        "completion_authoritative": False,
        "contracts_frozen": False,
        "duckdb_or_quack_state_written": False,
        "ordinary_runtime_instantiates_mock_worker": ordinary_instantiates,
        "mock_worker_quarantined": mock_quarantined,
        "workflow_coordinator_requires_explicit_simulation": workflow_gated,
        "simulated_results_represented_as_live": simulated_as_live,
        "live_cuda_qualified": False,
        "live_cuda_evidence_kind": "unavailable",
        "this_task_created_competing_authority": False,
        "probes": [item.to_mapping() for item in normalized],
        "blockers": list(dict.fromkeys(blockers)),
    }
    return QuarantineVerdict(
        schema=QUARANTINE_VERDICT_SCHEMA,
        interface=QUARANTINE_INTERFACE,
        promotion_status=promotion_status,
        supervisor_disposition="supervisor_non_promoted",
        closed_release_outcome=None,
        release_claim=False,
        completion_authoritative=False,
        contracts_frozen=False,
        duckdb_or_quack_state_written=False,
        ordinary_runtime_instantiates_mock_worker=ordinary_instantiates,
        mock_worker_quarantined=mock_quarantined,
        workflow_coordinator_requires_explicit_simulation=workflow_gated,
        simulated_results_represented_as_live=simulated_as_live,
        live_cuda_qualified=False,
        live_cuda_evidence_kind="unavailable",
        this_task_created_competing_authority=False,
        probes=tuple(normalized),
        blockers=tuple(dict.fromkeys(blockers)),
        verdict_cid=content_identity(payload),
    )


def qualify_current_head_quarantine() -> QuarantineVerdict:
    """Ordinary current-head PCPR-030 evaluation: honest R&D quarantine."""

    return qualify_legacy_mock_coordinator_quarantine(
        probes=current_head_quarantine_probes(),
    )


# Pinned identity of the ordinary current-head quarantine verdict.  Drift
# means the default payload changed and the outer receipt must be regenerated.
CURRENT_HEAD_NON_PROMOTION_VERDICT_CID: Final = (
    "baguqeerahe7vw4jogpnldm4n6kiflbky3ipurie6d6ndzsctfjtvwayqejta"
)


def pcpr_030_receipt_promotion(verdict: QuarantineVerdict) -> dict[str, Any]:
    """Compact outer-receipt promotion fields. Never a closed release."""

    if verdict.closed_release_outcome is not None:
        raise LegacyMockCoordinatorQuarantineError(
            "legacy mock-coordinator quarantine must not mint a closed release outcome"
        )
    if verdict.release_claim:
        raise LegacyMockCoordinatorQuarantineError(
            "legacy mock-coordinator quarantine must not claim a PCPR release"
        )
    if verdict.completion_authoritative:
        raise LegacyMockCoordinatorQuarantineError(
            "legacy mock-coordinator quarantine completion is not authoritative"
        )
    if verdict.duckdb_or_quack_state_written:
        raise LegacyMockCoordinatorQuarantineError(
            "legacy mock-coordinator quarantine must not write DuckDB or Quack state"
        )
    if verdict.promotion_status in CLOSED_RELEASE_OUTCOMES:
        raise LegacyMockCoordinatorQuarantineError(
            "promotion_status must not be a closed release outcome"
        )
    if verdict.promotion_status not in PROMOTION_STATUSES:
        raise LegacyMockCoordinatorQuarantineError(
            "promotion_status is not an admitted PCPR-030 status"
        )
    if verdict.promotion_status == "supervisor_promoted":
        raise LegacyMockCoordinatorQuarantineError(
            "legacy mock-coordinator quarantine cannot promote the supervisor"
        )
    if verdict.live_cuda_qualified:
        raise LegacyMockCoordinatorQuarantineError(
            "this task cannot claim live CUDA qualification"
        )
    return {
        "schema": QUARANTINE_VERDICT_SCHEMA,
        "interface": QUARANTINE_INTERFACE,
        "verdict_cid": verdict.verdict_cid,
        "promotion_status": verdict.promotion_status,
        "supervisor_disposition": verdict.supervisor_disposition,
        "closed_release_outcome": None,
        "release_claim": False,
        "completion_authoritative": False,
        "contracts_frozen": False,
        "duckdb_or_quack_state_written": False,
        "ordinary_runtime_instantiates_mock_worker": (
            verdict.ordinary_runtime_instantiates_mock_worker
        ),
        "mock_worker_quarantined": verdict.mock_worker_quarantined,
        "workflow_coordinator_requires_explicit_simulation": (
            verdict.workflow_coordinator_requires_explicit_simulation
        ),
        "simulated_results_represented_as_live": (
            verdict.simulated_results_represented_as_live
        ),
        "live_cuda_qualified": False,
        "live_cuda_evidence_kind": "unavailable",
        "this_task_created_competing_authority": (
            verdict.this_task_created_competing_authority
        ),
        "blocker_count": len(verdict.blockers),
        "blockers": list(verdict.blockers),
        "evidence_kind": "measured",
    }


def current_head_pcpr_030_receipt_promotion() -> dict[str, Any]:
    return pcpr_030_receipt_promotion(qualify_current_head_quarantine())


def pcpr_030_receipt_negative_results() -> dict[str, Any]:
    return {
        "simulated_presence_cannot_count_as_live": True,
        "simulated_results_not_represented_as_live": True,
        "estimated_values_rejected": True,
        "hermetic_pass_cannot_qualify_live": True,
        "closed_release_outcome_not_emitted": True,
        "direct_database_bypass_not_used": True,
        "live_cuda_not_claimed": True,
        "ordinary_runtime_cannot_instantiate_mock_worker": True,
        "evidence_kind": "measured",
    }


def current_head_pcpr_030_receipt_sections() -> dict[str, Any]:
    verdict = qualify_current_head_quarantine()
    promotion = pcpr_030_receipt_promotion(verdict)
    qualification = qualify_current_head_without_live_campaign()
    return {
        "qualification_verdict": promotion,
        "qualification_prerequisite": {
            "task_id": PCPR_003_TASK_ID,
            "goal_id": "PCPR-G130",
            "promotion_status": "rnd_non_promoted",
            "inventory_verdict_cid": PCPR_003_VERDICT_CID,
            "supervisor_live_qualification_task_id": PCPR_001_TASK_ID,
            "supervisor_live_qualification_promotion_status": qualification.promotion_status,
            "live_qualification_evidence_kind": "unavailable",
            "verdict_cid": CURRENT_HEAD_UNAVAILABLE_VERDICT_CID,
            "closed_release_outcome": None,
            "release_claim": False,
            "completion_authoritative": False,
            "reason": (
                "PCPR-003 inventoried the mock coordinator as present; this task "
                "quarantines it. PCPR-001 live qualification remains "
                "rnd_non_promoted and is not promoted by this receipt."
            ),
            "evidence_kind": "measured",
        },
        "quarantine": {
            "schema": QUARANTINE_SCHEMA,
            "interface": QUARANTINE_INTERFACE,
            "namespace": "ipfs_accelerate_py.compatibility.simulation.legacy_mock_coordinator",
            "ordinary_runtime_instantiates_mock_worker": (
                verdict.ordinary_runtime_instantiates_mock_worker
            ),
            "mock_worker_quarantined": verdict.mock_worker_quarantined,
            "workflow_coordinator_requires_explicit_simulation": (
                verdict.workflow_coordinator_requires_explicit_simulation
            ),
            "probes": [item.to_mapping() for item in verdict.probes],
            "evidence_kind": "measured",
        },
        "negative_results": pcpr_030_receipt_negative_results(),
        "verdict_cid": verdict.verdict_cid,
        "promotion_status": promotion["promotion_status"],
        "closed_release_outcome": None,
        "release_claim": False,
        "contracts_frozen": False,
    }


def pcpr_030_current_tree_binding(
    *,
    outer_commit: str,
    outer_tree: str,
    outer_subject: str,
    origin_main: str,
    origin_main_is_ancestor: bool,
    accelerator_pre_change_commit: str,
    accelerator_pre_change_tree: str,
    accelerator_gitlink: str,
    accelerator_origin_main: str,
    accelerator_origin_main_is_ancestor: bool,
    datasets_commit: str,
    datasets_tree: str,
    datasets_gitlink: str,
    kit_commit: str,
    kit_tree: str,
    kit_gitlink: str,
) -> dict[str, Any]:
    outer = _git_object_id(outer_commit, "outer_commit")
    tree = _git_object_id(outer_tree, "outer_tree")
    subject = _text(outer_subject, "outer_subject")
    origin = _git_object_id(origin_main, "origin_main")
    _require_ancestor(origin_main_is_ancestor, "origin_main_is_ancestor")
    accel = _git_object_id(accelerator_pre_change_commit, "accelerator_pre_change_commit")
    accel_tree = _git_object_id(accelerator_pre_change_tree, "accelerator_pre_change_tree")
    accel_link = _git_object_id(accelerator_gitlink, "accelerator_gitlink")
    accel_origin = _git_object_id(accelerator_origin_main, "accelerator_origin_main")
    _require_ancestor(
        accelerator_origin_main_is_ancestor, "accelerator_origin_main_is_ancestor"
    )
    if accel != accel_link:
        raise LegacyMockCoordinatorQuarantineError(
            "accelerator_pre_change_commit must equal accelerator_gitlink"
        )
    datasets = _git_object_id(datasets_commit, "datasets_commit")
    datasets_tree_id = _git_object_id(datasets_tree, "datasets_tree")
    datasets_link = _git_object_id(datasets_gitlink, "datasets_gitlink")
    if datasets != datasets_link:
        raise LegacyMockCoordinatorQuarantineError(
            "datasets_commit must equal datasets_gitlink"
        )
    kit = _git_object_id(kit_commit, "kit_commit")
    kit_tree_id = _git_object_id(kit_tree, "kit_tree")
    kit_link = _git_object_id(kit_gitlink, "kit_gitlink")
    if kit != kit_link:
        raise LegacyMockCoordinatorQuarantineError("kit_commit must equal kit_gitlink")
    _reject_closed_release_value(subject, "outer_subject")
    return {
        "outer_repository": "endomorphosis/lift_coding",
        "owning_repository_for_receipts": "ipfs_accelerate_py",
        "outer_commit": outer,
        "outer_tree": tree,
        "outer_subject": subject,
        "origin_main": origin,
        "origin_main_is_ancestor": True,
        "accelerator_pre_change_commit": accel,
        "accelerator_pre_change_tree": accel_tree,
        "accelerator_gitlink": accel_link,
        "accelerator_origin_main": accel_origin,
        "accelerator_origin_main_is_ancestor": True,
        "accelerator_post_change_commit": "pending nested commit after admission",
        "accelerator_post_change_tree": (
            "dirty-worktree; exact CID after accepted nested commit"
        ),
        "datasets_commit": datasets,
        "datasets_tree": datasets_tree_id,
        "datasets_gitlink": datasets_link,
        "kit_commit": kit,
        "kit_tree": kit_tree_id,
        "kit_gitlink": kit_link,
        "evidence_kind": "measured",
    }


def current_head_pcpr_030_current_tree_binding() -> dict[str, Any]:
    return pcpr_030_current_tree_binding(
        outer_commit=CURRENT_HEAD_OUTER_COMMIT,
        outer_tree=CURRENT_HEAD_OUTER_TREE,
        outer_subject=CURRENT_HEAD_OUTER_SUBJECT,
        origin_main=CURRENT_HEAD_ORIGIN_MAIN,
        origin_main_is_ancestor=True,
        accelerator_pre_change_commit=CURRENT_HEAD_ACCELERATOR_COMMIT,
        accelerator_pre_change_tree=CURRENT_HEAD_ACCELERATOR_TREE,
        accelerator_gitlink=CURRENT_HEAD_ACCELERATOR_COMMIT,
        accelerator_origin_main=CURRENT_HEAD_ACCELERATOR_ORIGIN_MAIN,
        accelerator_origin_main_is_ancestor=True,
        datasets_commit=CURRENT_HEAD_DATASETS_COMMIT,
        datasets_tree=CURRENT_HEAD_DATASETS_TREE,
        datasets_gitlink=CURRENT_HEAD_DATASETS_COMMIT,
        kit_commit=CURRENT_HEAD_KIT_COMMIT,
        kit_tree=CURRENT_HEAD_KIT_TREE,
        kit_gitlink=CURRENT_HEAD_KIT_COMMIT,
    )


def validate_pcpr_030_outer_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed if an outer PCPR-030 receipt claims a closed release."""

    if not isinstance(payload, Mapping):
        raise LegacyMockCoordinatorQuarantineError("outer receipt must be a mapping")
    if payload.get("task_id") != PCPR_030_TASK_ID:
        raise LegacyMockCoordinatorQuarantineError("outer receipt task_id must be PCPR-030")
    _reject_closed_release_value(payload.get("status"), "status")
    _reject_closed_release_value(payload.get("promotion_status"), "promotion_status")
    if payload.get("release_claim") is True:
        raise LegacyMockCoordinatorQuarantineError("outer receipt must not claim a release")
    if payload.get("completion_authoritative") is True:
        raise LegacyMockCoordinatorQuarantineError(
            "outer receipt completion is not authoritative"
        )

    verdict_section = payload.get("qualification_verdict")
    if not isinstance(verdict_section, Mapping):
        raise LegacyMockCoordinatorQuarantineError("qualification_verdict must be a mapping")
    _reject_closed_release_value(
        verdict_section.get("promotion_status"),
        "qualification_verdict.promotion_status",
    )
    if verdict_section.get("closed_release_outcome") is not None:
        raise LegacyMockCoordinatorQuarantineError(
            "qualification_verdict.closed_release_outcome must be null"
        )
    if verdict_section.get("release_claim") is True:
        raise LegacyMockCoordinatorQuarantineError(
            "qualification_verdict must not claim a release"
        )
    if verdict_section.get("duckdb_or_quack_state_written") is True:
        raise LegacyMockCoordinatorQuarantineError(
            "legacy mock-coordinator quarantine must not write DuckDB or Quack state"
        )
    if verdict_section.get("contracts_frozen") is True:
        raise LegacyMockCoordinatorQuarantineError(
            "legacy mock-coordinator quarantine cannot freeze contracts"
        )
    if verdict_section.get("live_cuda_qualified") is True:
        raise LegacyMockCoordinatorQuarantineError(
            "this task cannot claim live CUDA qualification"
        )
    promotion_status = verdict_section.get("promotion_status")
    if promotion_status not in PROMOTION_STATUSES:
        raise LegacyMockCoordinatorQuarantineError(
            "qualification_verdict.promotion_status is not an admitted PCPR-030 status"
        )
    if promotion_status == "supervisor_promoted":
        raise LegacyMockCoordinatorQuarantineError(
            "legacy mock-coordinator quarantine cannot promote the supervisor"
        )

    acceptance = payload.get("acceptance")
    if isinstance(acceptance, Mapping):
        _reject_closed_release_value(
            acceptance.get("promotion_status"), "acceptance.promotion_status"
        )
        if acceptance.get("closed_release_outcome") is not None:
            raise LegacyMockCoordinatorQuarantineError(
                "acceptance.closed_release_outcome must be null"
            )
        if acceptance.get("release_claim") is True:
            raise LegacyMockCoordinatorQuarantineError("acceptance must not claim a release")

    expected = current_head_pcpr_030_receipt_promotion()
    if verdict_section.get("verdict_cid") != expected["verdict_cid"]:
        raise LegacyMockCoordinatorQuarantineError(
            "qualification_verdict.verdict_cid must match the evaluator"
        )
    if promotion_status != expected["promotion_status"]:
        raise LegacyMockCoordinatorQuarantineError(
            "qualification_verdict.promotion_status must match the evaluator"
        )
    if expected["verdict_cid"] != CURRENT_HEAD_NON_PROMOTION_VERDICT_CID:
        raise LegacyMockCoordinatorQuarantineError(
            "pinned current-head non-promotion CID drifted from the evaluator"
        )

    return {
        "valid": True,
        "task_id": PCPR_030_TASK_ID,
        "promotion_status": promotion_status,
        "closed_release_outcome": None,
        "release_claim": False,
        "contracts_frozen": False,
        "verdict_cid": verdict_section.get("verdict_cid"),
        "evidence_kind": "measured",
    }


def file_digest(path: Path) -> tuple[str, int]:
    payload = path.read_bytes()
    return hashlib.sha256(payload).hexdigest(), len(payload)


__all__ = (
    "CLOSED_RELEASE_OUTCOMES",
    "CURRENT_HEAD_ACCELERATOR_COMMIT",
    "CURRENT_HEAD_NON_PROMOTION_VERDICT_CID",
    "HERMETIC_CANDIDATE_SUITES",
    "PCPR_030_GOAL_ID",
    "PCPR_030_TASK_ID",
    "QUARANTINE_INTERFACE",
    "SEALED_GIT_BINARY",
    "SEALED_PATH",
    "SEALED_PYTHON",
    "LegacyMockCoordinatorQuarantineError",
    "QuarantineProbe",
    "QuarantineVerdict",
    "current_head_pcpr_030_current_tree_binding",
    "current_head_pcpr_030_receipt_promotion",
    "current_head_pcpr_030_receipt_sections",
    "current_head_quarantine_probes",
    "pcpr_030_current_tree_binding",
    "pcpr_030_receipt_promotion",
    "qualify_current_head_quarantine",
    "qualify_legacy_mock_coordinator_quarantine",
    "validate_pcpr_030_outer_receipt",
)
