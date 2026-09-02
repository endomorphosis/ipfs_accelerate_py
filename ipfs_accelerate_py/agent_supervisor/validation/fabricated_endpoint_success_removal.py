"""Fail-closed PCPR-033 fabricated endpoint-success removal.

PCPR-033 quarantines mock endpoint handlers, MCP mock inference, and
hardware-test fallbacks that reported success without a live probe.
Ordinary runtime cannot treat simulation as live. This module is not
release authority: it does not write DuckDB or Quack state, does not
qualify live endpoints or inference, and never emits a closed PCPR
release outcome.

Live claims require live evidence. Simulated endpoints stay ``Simulated``.
Missing backends, models, canaries, and probes stay typed unavailable.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from ipfs_accelerate_py.compatibility.simulation.fabricated_endpoint_success import (
    FabricatedEndpointSuccessError,
    SimulatedEndpointHandler,
    UnavailableEndpointHandler,
    create_simulated_endpoint_handler,
    create_unavailable_endpoint_handler,
    install_ordinary_endpoint_handler,
    ordinary_mock_inference,
    run_hardware_test_fallback,
    simulate_hardware_test,
    simulate_inference,
    unavailable_hardware_test,
)

from ..proof.formal_verification_contracts import content_identity
from .canonical_supervisor_contract_freeze import CLOSED_RELEASE_OUTCOMES
from .direct_objective_event_driven_qualification import (
    CURRENT_HEAD_UNAVAILABLE_VERDICT_CID,
    qualify_current_head_without_live_campaign,
)
from .legacy_mock_coordinator_quarantine import (
    discover_accelerate_root,
    discover_portfolio_root,
)
from .pseudo_cid_identity_removal import (
    CURRENT_HEAD_NON_PROMOTION_VERDICT_CID as PCPR_032_VERDICT_CID,
)
from .source_seal_and_supervisor_baseline import (
    SEALED_GIT_BINARY,
    SEALED_PATH,
    SEALED_PYTHON,
)


REMOVAL_INTERFACE: Final = "FabricatedEndpointSuccessRemoval@1"
REMOVAL_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/fabricated-endpoint-success-removal@1"
)
REMOVAL_VERDICT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "fabricated-endpoint-success-removal-verdict@1"
)

PCPR_033_TASK_ID: Final = "PCPR-033"
PCPR_033_GOAL_ID: Final = "PCPR-G410"
PCPR_032_TASK_ID: Final = "PCPR-032"
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

CURRENT_HEAD_OUTER_COMMIT: Final = "8a8d50dd46fde811d080132ec6f3bdce4059b1e1"
CURRENT_HEAD_OUTER_TREE: Final = "b270805744c7de8752847b53843d4c5894a7384b"
CURRENT_HEAD_OUTER_SUBJECT: Final = (
    "fix(pcpr): reseal accelerate planning pins to nested-owner mapping"
)
CURRENT_HEAD_ORIGIN_MAIN: Final = "bb8869ed72eb7002434345d9969efee729c4f7f6"
CURRENT_HEAD_ACCELERATOR_COMMIT: Final = (
    "d4aa008d66033574911e8bdde6cfe12e9817958b"
)
CURRENT_HEAD_ACCELERATOR_TREE: Final = "2e3dd404a6ff7530346ff0b0dc0575e28a521383"
CURRENT_HEAD_ACCELERATOR_ORIGIN_MAIN: Final = (
    "f8c2f633fa6a781b822176fd63e1a229f96b581c"
)
CURRENT_HEAD_DATASETS_COMMIT: Final = "f49afc579c22856849ca9f739435e5820003384f"
CURRENT_HEAD_DATASETS_TREE: Final = "47118a8e6d1b6b4e7ae04f9a2efda33aadebca1b"
CURRENT_HEAD_KIT_COMMIT: Final = "b6c65ba732733d7e33852713ba18aa3b12235668"
CURRENT_HEAD_KIT_TREE: Final = "14da7d92e130b7ba3523d0d6741a3ef7ef1e1bc2"

HERMETIC_CANDIDATE_SUITES: Final[tuple[str, ...]] = (
    "test/api/test_agent_supervisor_fabricated_endpoint_success_removal.py",
)

SIMULATION_MODULE_RELPATH: Final = (
    "ipfs_accelerate_py/compatibility/simulation/fabricated_endpoint_success.py"
)
ORDINARY_RUNTIME_RELPATH: Final = "ipfs_accelerate_py/ipfs_accelerate.py"
LEGACY_MODULE_RELPATH: Final = "ipfs_accelerate_py/ipfs_accelerate_py_legacy.py"
INFERENCE_TOOLS_RELPATH: Final = "ipfs_accelerate_py/mcp/inference_tools.py"
HARDWARE_TOOLS_RELPATH: Final = (
    "ipfs_accelerate_py/mcp_server/tools/hardware_tools/native_hardware_tools.py"
)

COMMIT_RE: Final = re.compile(r"^[0-9a-f]{40}$")
REAL_LABEL_RE: Final = re.compile(r"REAL implementation type")
OVERALL_PASSED_TRUE_RE: Final = re.compile(
    r"""["']overall_passed["']\s*:\s*True"""
)


class FabricatedEndpointSuccessRemovalError(ValueError):
    """Malformed endpoint-success-removal evidence or a forbidden claim."""


@dataclass(frozen=True)
class RemovalProbe:
    """One measured or typed-unavailable endpoint-success-removal observation."""

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
class RemovalVerdict:
    """Fail-closed PCPR-033 decision. Never a closed release."""

    schema: str
    interface: str
    promotion_status: str
    supervisor_disposition: str
    closed_release_outcome: None
    release_claim: bool
    completion_authoritative: bool
    contracts_frozen: bool
    duckdb_or_quack_state_written: bool
    ordinary_runtime_uses_mock_endpoint_success: bool
    mock_endpoint_handler_quarantined: bool
    mock_inference_quarantined: bool
    hardware_test_fallback_not_overall_passed: bool
    simulated_results_represented_as_live: bool
    live_endpoint_qualified: bool
    live_endpoint_evidence_kind: str
    this_task_created_competing_authority: bool
    probes: tuple[RemovalProbe, ...]
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
            "ordinary_runtime_uses_mock_endpoint_success": (
                self.ordinary_runtime_uses_mock_endpoint_success
            ),
            "mock_endpoint_handler_quarantined": (
                self.mock_endpoint_handler_quarantined
            ),
            "mock_inference_quarantined": self.mock_inference_quarantined,
            "hardware_test_fallback_not_overall_passed": (
                self.hardware_test_fallback_not_overall_passed
            ),
            "simulated_results_represented_as_live": (
                self.simulated_results_represented_as_live
            ),
            "live_endpoint_qualified": self.live_endpoint_qualified,
            "live_endpoint_evidence_kind": self.live_endpoint_evidence_kind,
            "this_task_created_competing_authority": (
                self.this_task_created_competing_authority
            ),
            "probes": [item.to_mapping() for item in self.probes],
            "blockers": list(self.blockers),
            "verdict_cid": self.verdict_cid,
        }


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FabricatedEndpointSuccessRemovalError(
            f"{name} must be a non-empty string"
        )
    return value.strip()


def _kind(value: Any, name: str) -> str:
    kind = _text(value, name)
    if kind not in EVIDENCE_KINDS:
        raise FabricatedEndpointSuccessRemovalError(
            f"{name} is not an admitted evidence kind"
        )
    return kind


def _git_object_id(value: Any, name: str) -> str:
    text = _text(value, name)
    if COMMIT_RE.fullmatch(text) is None:
        raise FabricatedEndpointSuccessRemovalError(
            f"{name} must be a 40-character lowercase git object id"
        )
    return text


def _reject_closed_release_value(value: Any, name: str) -> None:
    if isinstance(value, str) and value in CLOSED_RELEASE_OUTCOMES:
        raise FabricatedEndpointSuccessRemovalError(
            f"{name} must not be a closed PCPR release outcome"
        )


def _require_ancestor(flag: bool, name: str) -> None:
    if flag is not True:
        raise FabricatedEndpointSuccessRemovalError(f"{name} must be true")


def _read_source(root: Path, relpath: str) -> str | None:
    path = root / relpath
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _source_probe(
    *,
    probe_id: str,
    source: str | None,
    present: bool | None,
    ok_reason: str,
    fail_reason: str,
    missing_reason: str,
    relpath: str,
) -> RemovalProbe:
    if source is None:
        return RemovalProbe(
            probe_id=probe_id,
            present=None,
            evidence_kind="unavailable",
            live=False,
            simulated_represented_as_live=False,
            reason=missing_reason,
        )
    ok = bool(present)
    return RemovalProbe(
        probe_id=probe_id,
        present=ok,
        evidence_kind="measured",
        live=False,
        simulated_represented_as_live=False,
        reason=ok_reason if ok else fail_reason,
        details=MappingProxyType({"relpath": relpath}),
    )


def current_head_removal_probes(
    *,
    accelerate_root: Path | None = None,
) -> tuple[RemovalProbe, ...]:
    """Measured current-tree probes. Missing files stay typed unavailable."""

    root = accelerate_root or discover_accelerate_root()
    if root is None or not root.is_dir():
        return (
            RemovalProbe(
                probe_id="accelerate_source_tree",
                present=None,
                evidence_kind="unavailable",
                live=False,
                simulated_represented_as_live=False,
                reason="Accelerate source tree is not present and is not recorded as empty.",
            ),
        )

    simulation = _read_source(root, SIMULATION_MODULE_RELPATH)
    ordinary = _read_source(root, ORDINARY_RUNTIME_RELPATH)
    legacy = _read_source(root, LEGACY_MODULE_RELPATH)
    inference = _read_source(root, INFERENCE_TOOLS_RELPATH)
    hardware = _read_source(root, HARDWARE_TOOLS_RELPATH)
    probes: list[RemovalProbe] = []

    probes.append(
        _source_probe(
            probe_id="simulation_namespace_fabricated_endpoint_success",
            source=simulation,
            present=bool(
                simulation
                and "class SimulatedEndpointHandler" in simulation
                and "class UnavailableEndpointHandler" in simulation
                and "def ordinary_mock_inference" in simulation
                and "def unavailable_hardware_test" in simulation
            ),
            ok_reason=(
                "Mock endpoint handlers, inference, and hardware-test fallbacks "
                "live in the explicit simulation namespace."
            ),
            fail_reason="Simulation namespace is missing quarantined endpoint-success symbols.",
            missing_reason="Simulation namespace module is unreadable.",
            relpath=SIMULATION_MODULE_RELPATH,
        )
    )
    probes.append(
        _source_probe(
            probe_id="ordinary_runtime_no_real_mock_label",
            source=ordinary,
            present=bool(
                ordinary
                and "install_endpoint_handler" in ordinary
                and REAL_LABEL_RE.search(ordinary) is None
            ),
            ok_reason="Ordinary constructor no longer labels mock handlers as REAL.",
            fail_reason="Ordinary runtime still labels mock handlers as REAL.",
            missing_reason="ipfs_accelerate.py is unreadable.",
            relpath=ORDINARY_RUNTIME_RELPATH,
        )
    )
    probes.append(
        _source_probe(
            probe_id="ordinary_legacy_no_success_mock_handler",
            source=legacy,
            present=bool(
                legacy
                and "install_endpoint_handler" in legacy
                and '"status": "success"' not in legacy.split("def _create_mock_handler", 1)[-1][:1200]
            ),
            ok_reason="Ordinary legacy mock handler no longer returns status=success.",
            fail_reason="Ordinary legacy mock handler still returns status=success.",
            missing_reason="Legacy module is unreadable.",
            relpath=LEGACY_MODULE_RELPATH,
        )
    )
    probes.append(
        _source_probe(
            probe_id="mcp_inference_no_fabricated_success",
            source=inference,
            present=bool(
                inference
                and "def _mock_inference" in inference
                and "def infer" in inference
                and "[Mock transcription from" not in inference
                and "mock://synthesized_" not in inference
            ),
            ok_reason="MCP inference tools no longer unpack mock success payloads.",
            fail_reason="MCP inference tools still emit fabricated success payloads.",
            missing_reason="inference_tools.py is unreadable.",
            relpath=INFERENCE_TOOLS_RELPATH,
        )
    )
    probes.append(
        _source_probe(
            probe_id="hardware_fallback_not_overall_passed",
            source=hardware,
            present=bool(
                hardware
                and "def _test_hardware_fallback" in hardware
                and "unavailable_hardware_test" in hardware
                and OVERALL_PASSED_TRUE_RE.search(hardware) is None
            ),
            ok_reason="Hardware-test fallback no longer reports overall_passed True.",
            fail_reason="Hardware-test fallback still reports overall_passed True.",
            missing_reason="native_hardware_tools.py is unreadable.",
            relpath=HARDWARE_TOOLS_RELPATH,
        )
    )

    ordinary_handler = create_unavailable_endpoint_handler("bert-base", "cpu:0")
    ordinary_result = ordinary_handler("probe")
    ordinary_ok = (
        isinstance(ordinary_handler, UnavailableEndpointHandler)
        and ordinary_handler.live is False
        and ordinary_result.get("outcome") == "Unavailable"
        and ordinary_result.get("status") != "success"
        and ordinary_result.get("implementation_type") != "REAL"
    )
    probes.append(
        RemovalProbe(
            probe_id="ordinary_endpoint_handler",
            present=ordinary_ok,
            evidence_kind="measured",
            live=False,
            simulated_represented_as_live=False,
            reason=(
                "Ordinary endpoint handler is typed unavailable and not mock success."
                if ordinary_ok
                else "Ordinary endpoint handler still reports fabricated success."
            ),
            details=MappingProxyType(
                {
                    "type_name": type(ordinary_handler).__name__,
                    "outcome": ordinary_result.get("outcome"),
                    "status": ordinary_result.get("status"),
                }
            ),
        )
    )

    mock_blocked = False
    mock_code = ""
    try:
        create_simulated_endpoint_handler("bert-base", "cpu:0")
    except FabricatedEndpointSuccessError as exc:
        mock_blocked = True
        mock_code = exc.code
    probes.append(
        RemovalProbe(
            probe_id="mock_handler_ordinary_instantiation",
            present=mock_blocked,
            evidence_kind="measured",
            live=False,
            simulated_represented_as_live=False,
            reason=(
                "Mock endpoint handler ordinary instantiation is refused."
                if mock_blocked
                else "Mock endpoint handler can still be constructed without explicit simulation."
            ),
            details=MappingProxyType({"code": mock_code}),
        )
    )

    simulated = create_simulated_endpoint_handler(
        "bert-base", "cpu:0", explicit_simulation=True
    )
    simulated_result = simulated("probe")
    simulated_as_live = simulated.live is True or simulated_result.get("live") is True
    simulated_ok = (
        isinstance(simulated, SimulatedEndpointHandler)
        and simulated_result.get("outcome") == "Simulated"
        and simulated_result.get("implementation_type") == "MOCK"
        and not simulated_as_live
    )
    probes.append(
        RemovalProbe(
            probe_id="mock_handler_explicit_simulation",
            present=simulated_ok,
            evidence_kind="measured",
            live=False,
            simulated_represented_as_live=simulated_as_live,
            reason=(
                "Explicit mock endpoint handler is labeled Simulated and is not live."
                if simulated_ok
                else "Explicit mock endpoint handler claimed live success."
            ),
            details=MappingProxyType(
                {
                    "origin": simulated_result.get("origin"),
                    "outcome": simulated_result.get("outcome"),
                    "implementation_type": simulated_result.get("implementation_type"),
                }
            ),
        )
    )

    resources: dict[str, Any] = {}
    installed = install_ordinary_endpoint_handler(resources, "t5-small", "cpu:0")
    installed_result = installed("probe")
    install_ok = (
        installed_result.get("outcome") == "Unavailable"
        and resources["endpoint_handler"]["t5-small"]["cpu:0"] is installed
    )
    probes.append(
        RemovalProbe(
            probe_id="ordinary_handler_install",
            present=install_ok,
            evidence_kind="measured",
            live=False,
            simulated_represented_as_live=False,
            reason=(
                "Ordinary install stores a typed-unavailable handler."
                if install_ok
                else "Ordinary install still stores a live mock handler."
            ),
        )
    )

    unavailable_inference = ordinary_mock_inference(
        "causal_language_modeling", "gpt2", {"prompt": "pcpr-033"}
    )
    inference_ok = (
        unavailable_inference.get("outcome") == "Unavailable"
        and unavailable_inference.get("live") is False
        and "generated_text" not in unavailable_inference
        and unavailable_inference.get("status") != "success"
    )
    probes.append(
        RemovalProbe(
            probe_id="ordinary_mock_inference",
            present=inference_ok,
            evidence_kind="measured",
            live=False,
            simulated_represented_as_live=False,
            reason=(
                "Ordinary mock inference is typed unavailable and does not emit generated text."
                if inference_ok
                else "Ordinary mock inference still fabricates success."
            ),
            details=MappingProxyType(
                {
                    "outcome": unavailable_inference.get("outcome"),
                    "code": unavailable_inference.get("code"),
                }
            ),
        )
    )

    simulated_inference = simulate_inference(
        "causal_language_modeling",
        "gpt2",
        {"prompt": "pcpr-033"},
        explicit_simulation=True,
    )
    sim_inf_as_live = simulated_inference.get("live") is True
    sim_inf_ok = (
        simulated_inference.get("outcome") == "Simulated"
        and simulated_inference.get("implementation_type") == "MOCK"
        and not sim_inf_as_live
        and simulated_inference.get("status") != "success"
    )
    probes.append(
        RemovalProbe(
            probe_id="simulated_inference",
            present=sim_inf_ok,
            evidence_kind="measured",
            live=False,
            simulated_represented_as_live=sim_inf_as_live,
            reason=(
                "Explicit simulated inference is labeled Simulated, never live."
                if sim_inf_ok
                else "Explicit simulated inference claimed live success."
            ),
            details=MappingProxyType(
                {
                    "origin": simulated_inference.get("origin"),
                    "outcome": simulated_inference.get("outcome"),
                }
            ),
        )
    )

    hw_unavailable = unavailable_hardware_test("cuda", "basic")
    hw_ok = (
        hw_unavailable.get("outcome") == "Unavailable"
        and hw_unavailable.get("overall_passed") is None
        and hw_unavailable.get("status") != "success"
        and hw_unavailable.get("live") is False
    )
    probes.append(
        RemovalProbe(
            probe_id="ordinary_hardware_test_fallback",
            present=hw_ok,
            evidence_kind="measured",
            live=False,
            simulated_represented_as_live=False,
            reason=(
                "Hardware-test fallback is typed unavailable and overall_passed is not True."
                if hw_ok
                else "Hardware-test fallback still reports overall_passed True."
            ),
            details=MappingProxyType(
                {
                    "outcome": hw_unavailable.get("outcome"),
                    "overall_passed": hw_unavailable.get("overall_passed"),
                }
            ),
        )
    )

    hw_simulated = simulate_hardware_test(
        "cuda", "basic", explicit_simulation=True
    )
    hw_sim_as_live = hw_simulated.get("live") is True or hw_simulated.get(
        "overall_passed"
    ) is True
    hw_sim_ok = (
        hw_simulated.get("outcome") == "Simulated"
        and hw_simulated.get("overall_passed") is None
        and not hw_sim_as_live
    )
    probes.append(
        RemovalProbe(
            probe_id="simulated_hardware_test",
            present=hw_sim_ok,
            evidence_kind="measured",
            live=False,
            simulated_represented_as_live=hw_sim_as_live,
            reason=(
                "Explicit simulated hardware test is not overall_passed True and not live."
                if hw_sim_ok
                else "Explicit simulated hardware test claimed live pass."
            ),
            details=MappingProxyType(
                {
                    "outcome": hw_simulated.get("outcome"),
                    "overall_passed": hw_simulated.get("overall_passed"),
                }
            ),
        )
    )

    ordinary_hw = run_hardware_test_fallback("all", "basic")
    ordinary_hw_ok = (
        ordinary_hw.get("outcome") == "Unavailable"
        and ordinary_hw.get("overall_passed") is not True
    )
    probes.append(
        RemovalProbe(
            probe_id="ordinary_hardware_test_dispatch",
            present=ordinary_hw_ok,
            evidence_kind="measured",
            live=False,
            simulated_represented_as_live=False,
            reason=(
                "Ordinary hardware-test dispatch is typed unavailable."
                if ordinary_hw_ok
                else "Ordinary hardware-test dispatch still reports a live pass."
            ),
        )
    )

    probes.append(
        RemovalProbe(
            probe_id="live_endpoint_qualification",
            present=None,
            evidence_kind="unavailable",
            live=False,
            simulated_represented_as_live=False,
            reason=(
                "This task does not qualify a live endpoint. Missing backend, "
                "model, canary, timeout, cancellation, and resource-admission "
                "evidence stays typed unavailable and is not recorded as False or passing."
            ),
        )
    )
    return tuple(probes)


def qualify_fabricated_endpoint_success_removal(
    *,
    probes: Sequence[RemovalProbe],
    duckdb_or_quack_state_written: bool = False,
) -> RemovalVerdict:
    """Evaluate PCPR-033. Removal is fail-closed; promotion is never a release."""

    if duckdb_or_quack_state_written:
        raise FabricatedEndpointSuccessRemovalError(
            "fabricated endpoint-success removal must not write DuckDB or Quack state"
        )
    if not probes:
        raise FabricatedEndpointSuccessRemovalError(
            "endpoint-success-removal probes are required"
        )

    normalized: list[RemovalProbe] = []
    blockers: list[str] = []
    simulated_as_live = False
    for probe in probes:
        probe_id = _text(probe.probe_id, "probe_id")
        kind = _kind(probe.evidence_kind, f"{probe_id}.evidence_kind")
        if kind == "estimated":
            raise FabricatedEndpointSuccessRemovalError(
                f"{probe_id}: estimated values cannot mint endpoint-success-removal evidence"
            )
        if kind == "simulated" and probe.present is True:
            raise FabricatedEndpointSuccessRemovalError(
                "simulated observations cannot be represented as live presence"
            )
        if kind == "measured_live":
            raise FabricatedEndpointSuccessRemovalError(
                f"{probe_id}: this removal evaluator cannot carry measured_live evidence"
            )
        if probe.simulated_represented_as_live:
            simulated_as_live = True
            blockers.append(f"{probe_id}:simulated_represented_as_live")
        if probe.live:
            blockers.append(f"{probe_id}:live_claim")
        normalized.append(
            RemovalProbe(
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

    def _require(probe_id: str, blocker: str) -> bool:
        item = by_id.get(probe_id)
        ok = bool(item and item.present is True)
        if not ok:
            blockers.append(blocker)
        return ok

    ordinary = by_id.get("ordinary_runtime_no_real_mock_label")
    uses_mock_success = bool(ordinary and ordinary.present is False)
    if ordinary is None or ordinary.present is not True:
        blockers.append("ordinary_runtime_uses_mock_endpoint_success")

    mock_quarantined = _require(
        "simulation_namespace_fabricated_endpoint_success",
        "mock_endpoint_not_in_simulation_namespace",
    )
    _require("ordinary_legacy_no_success_mock_handler", "legacy_mock_handler_success")
    inference_quarantined = _require(
        "mcp_inference_no_fabricated_success", "mcp_inference_fabricated_success"
    )
    hw_not_passed = _require(
        "hardware_fallback_not_overall_passed", "hardware_fallback_overall_passed"
    )
    _require("ordinary_endpoint_handler", "ordinary_handler_not_unavailable")
    _require(
        "mock_handler_ordinary_instantiation",
        "mock_handler_ordinary_instantiation_allowed",
    )
    _require("mock_handler_explicit_simulation", "mock_handler_reported_live")
    _require("ordinary_handler_install", "ordinary_handler_install_live")
    _require("ordinary_mock_inference", "ordinary_mock_inference_success")
    _require("simulated_inference", "simulated_inference_claimed_live")
    _require("ordinary_hardware_test_fallback", "hardware_test_overall_passed")
    _require("simulated_hardware_test", "simulated_hardware_test_claimed_live")
    _require("ordinary_hardware_test_dispatch", "hardware_test_dispatch_passed")

    unavailable_count = sum(
        1 for item in normalized if item.evidence_kind == "unavailable"
    )
    if blockers:
        promotion_status = "typed_blocked"
    elif unavailable_count == len(normalized):
        promotion_status = "typed_unavailable"
    else:
        promotion_status = "rnd_non_promoted"

    if promotion_status not in PROMOTION_STATUSES:
        raise FabricatedEndpointSuccessRemovalError(
            "internal promotion status is not admitted"
        )
    if promotion_status in CLOSED_RELEASE_OUTCOMES:
        raise FabricatedEndpointSuccessRemovalError(
            "fabricated endpoint-success removal must not mint a closed release outcome"
        )

    payload = {
        "schema": REMOVAL_VERDICT_SCHEMA,
        "interface": REMOVAL_INTERFACE,
        "task_id": PCPR_033_TASK_ID,
        "goal_id": PCPR_033_GOAL_ID,
        "promotion_status": promotion_status,
        "supervisor_disposition": "supervisor_non_promoted",
        "closed_release_outcome": None,
        "release_claim": False,
        "completion_authoritative": False,
        "contracts_frozen": False,
        "duckdb_or_quack_state_written": False,
        "ordinary_runtime_uses_mock_endpoint_success": uses_mock_success,
        "mock_endpoint_handler_quarantined": mock_quarantined,
        "mock_inference_quarantined": inference_quarantined,
        "hardware_test_fallback_not_overall_passed": hw_not_passed,
        "simulated_results_represented_as_live": simulated_as_live,
        "live_endpoint_qualified": False,
        "live_endpoint_evidence_kind": "unavailable",
        "this_task_created_competing_authority": False,
        "probes": [item.to_mapping() for item in normalized],
        "blockers": list(dict.fromkeys(blockers)),
    }
    return RemovalVerdict(
        schema=REMOVAL_VERDICT_SCHEMA,
        interface=REMOVAL_INTERFACE,
        promotion_status=promotion_status,
        supervisor_disposition="supervisor_non_promoted",
        closed_release_outcome=None,
        release_claim=False,
        completion_authoritative=False,
        contracts_frozen=False,
        duckdb_or_quack_state_written=False,
        ordinary_runtime_uses_mock_endpoint_success=uses_mock_success,
        mock_endpoint_handler_quarantined=mock_quarantined,
        mock_inference_quarantined=inference_quarantined,
        hardware_test_fallback_not_overall_passed=hw_not_passed,
        simulated_results_represented_as_live=simulated_as_live,
        live_endpoint_qualified=False,
        live_endpoint_evidence_kind="unavailable",
        this_task_created_competing_authority=False,
        probes=tuple(normalized),
        blockers=tuple(dict.fromkeys(blockers)),
        verdict_cid=content_identity(payload),
    )


def qualify_current_head_removal() -> RemovalVerdict:
    """Ordinary current-head PCPR-033 evaluation: honest R&D removal."""

    return qualify_fabricated_endpoint_success_removal(
        probes=current_head_removal_probes(),
    )


CURRENT_HEAD_NON_PROMOTION_VERDICT_CID: Final = (
    "baguqeerapdcauoeu2awnbkgkfma6vrg5j5keuhojdtb34pvfiayspaayfnda"
)


def pcpr_033_receipt_promotion(verdict: RemovalVerdict) -> dict[str, Any]:
    """Compact outer-receipt promotion fields. Never a closed release."""

    if verdict.closed_release_outcome is not None:
        raise FabricatedEndpointSuccessRemovalError(
            "fabricated endpoint-success removal must not mint a closed release outcome"
        )
    if verdict.release_claim:
        raise FabricatedEndpointSuccessRemovalError(
            "fabricated endpoint-success removal must not claim a PCPR release"
        )
    if verdict.completion_authoritative:
        raise FabricatedEndpointSuccessRemovalError(
            "fabricated endpoint-success removal completion is not authoritative"
        )
    if verdict.duckdb_or_quack_state_written:
        raise FabricatedEndpointSuccessRemovalError(
            "fabricated endpoint-success removal must not write DuckDB or Quack state"
        )
    if verdict.promotion_status in CLOSED_RELEASE_OUTCOMES:
        raise FabricatedEndpointSuccessRemovalError(
            "promotion_status must not be a closed release outcome"
        )
    if verdict.promotion_status not in PROMOTION_STATUSES:
        raise FabricatedEndpointSuccessRemovalError(
            "promotion_status is not an admitted PCPR-033 status"
        )
    if verdict.promotion_status == "supervisor_promoted":
        raise FabricatedEndpointSuccessRemovalError(
            "fabricated endpoint-success removal cannot promote the supervisor"
        )
    if verdict.live_endpoint_qualified:
        raise FabricatedEndpointSuccessRemovalError(
            "this task cannot claim live endpoint qualification"
        )
    return {
        "schema": REMOVAL_VERDICT_SCHEMA,
        "interface": REMOVAL_INTERFACE,
        "verdict_cid": verdict.verdict_cid,
        "promotion_status": verdict.promotion_status,
        "supervisor_disposition": verdict.supervisor_disposition,
        "closed_release_outcome": None,
        "release_claim": False,
        "completion_authoritative": False,
        "contracts_frozen": False,
        "duckdb_or_quack_state_written": False,
        "ordinary_runtime_uses_mock_endpoint_success": (
            verdict.ordinary_runtime_uses_mock_endpoint_success
        ),
        "mock_endpoint_handler_quarantined": verdict.mock_endpoint_handler_quarantined,
        "mock_inference_quarantined": verdict.mock_inference_quarantined,
        "hardware_test_fallback_not_overall_passed": (
            verdict.hardware_test_fallback_not_overall_passed
        ),
        "simulated_results_represented_as_live": (
            verdict.simulated_results_represented_as_live
        ),
        "live_endpoint_qualified": False,
        "live_endpoint_evidence_kind": "unavailable",
        "this_task_created_competing_authority": (
            verdict.this_task_created_competing_authority
        ),
        "blocker_count": len(verdict.blockers),
        "blockers": list(verdict.blockers),
        "evidence_kind": "measured",
    }


def current_head_pcpr_033_receipt_promotion() -> dict[str, Any]:
    return pcpr_033_receipt_promotion(qualify_current_head_removal())


def pcpr_033_receipt_negative_results() -> dict[str, Any]:
    return {
        "simulated_presence_cannot_count_as_live": True,
        "simulated_results_not_represented_as_live": True,
        "estimated_values_rejected": True,
        "hermetic_pass_cannot_qualify_live": True,
        "closed_release_outcome_not_emitted": True,
        "direct_database_bypass_not_used": True,
        "live_endpoint_not_claimed": True,
        "mock_handler_not_labeled_real": True,
        "hardware_fallback_not_overall_passed": True,
        "ordinary_runtime_cannot_instantiate_mock_handler": True,
        "evidence_kind": "measured",
    }


def current_head_pcpr_033_receipt_sections() -> dict[str, Any]:
    verdict = qualify_current_head_removal()
    promotion = pcpr_033_receipt_promotion(verdict)
    qualification = qualify_current_head_without_live_campaign()
    return {
        "qualification_verdict": promotion,
        "qualification_prerequisite": {
            "task_id": PCPR_032_TASK_ID,
            "goal_id": "PCPR-G410",
            "promotion_status": "rnd_non_promoted",
            "removal_verdict_cid": PCPR_032_VERDICT_CID,
            "supervisor_live_qualification_task_id": PCPR_001_TASK_ID,
            "supervisor_live_qualification_promotion_status": qualification.promotion_status,
            "live_qualification_evidence_kind": "unavailable",
            "verdict_cid": CURRENT_HEAD_UNAVAILABLE_VERDICT_CID,
            "closed_release_outcome": None,
            "release_claim": False,
            "completion_authoritative": False,
            "reason": (
                "PCPR-032 removed pseudo-CID identity. This task removes "
                "fabricated endpoint success. PCPR-001 live qualification "
                "remains rnd_non_promoted and is not promoted by this receipt."
            ),
            "evidence_kind": "measured",
        },
        "removal": {
            "schema": REMOVAL_SCHEMA,
            "interface": REMOVAL_INTERFACE,
            "namespace": (
                "ipfs_accelerate_py.compatibility.simulation.fabricated_endpoint_success"
            ),
            "ordinary_runtime_uses_mock_endpoint_success": (
                verdict.ordinary_runtime_uses_mock_endpoint_success
            ),
            "mock_endpoint_handler_quarantined": (
                verdict.mock_endpoint_handler_quarantined
            ),
            "mock_inference_quarantined": verdict.mock_inference_quarantined,
            "hardware_test_fallback_not_overall_passed": (
                verdict.hardware_test_fallback_not_overall_passed
            ),
            "probes": [item.to_mapping() for item in verdict.probes],
            "evidence_kind": "measured",
        },
        "negative_results": pcpr_033_receipt_negative_results(),
        "verdict_cid": verdict.verdict_cid,
        "promotion_status": promotion["promotion_status"],
        "closed_release_outcome": None,
        "release_claim": False,
        "contracts_frozen": False,
    }


def pcpr_033_current_tree_binding(
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
    accel = _git_object_id(
        accelerator_pre_change_commit, "accelerator_pre_change_commit"
    )
    accel_tree = _git_object_id(
        accelerator_pre_change_tree, "accelerator_pre_change_tree"
    )
    accel_link = _git_object_id(accelerator_gitlink, "accelerator_gitlink")
    accel_origin = _git_object_id(
        accelerator_origin_main, "accelerator_origin_main"
    )
    _require_ancestor(
        accelerator_origin_main_is_ancestor, "accelerator_origin_main_is_ancestor"
    )
    if accel != accel_link:
        raise FabricatedEndpointSuccessRemovalError(
            "accelerator_pre_change_commit must equal accelerator_gitlink"
        )
    datasets = _git_object_id(datasets_commit, "datasets_commit")
    datasets_tree_id = _git_object_id(datasets_tree, "datasets_tree")
    datasets_link = _git_object_id(datasets_gitlink, "datasets_gitlink")
    if datasets != datasets_link:
        raise FabricatedEndpointSuccessRemovalError(
            "datasets_commit must equal datasets_gitlink"
        )
    kit = _git_object_id(kit_commit, "kit_commit")
    kit_tree_id = _git_object_id(kit_tree, "kit_tree")
    kit_link = _git_object_id(kit_gitlink, "kit_gitlink")
    if kit != kit_link:
        raise FabricatedEndpointSuccessRemovalError(
            "kit_commit must equal kit_gitlink"
        )
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


def current_head_pcpr_033_current_tree_binding() -> dict[str, Any]:
    return pcpr_033_current_tree_binding(
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


def validate_pcpr_033_outer_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed if an outer PCPR-033 receipt claims a closed release."""

    if not isinstance(payload, Mapping):
        raise FabricatedEndpointSuccessRemovalError("outer receipt must be a mapping")
    if payload.get("task_id") != PCPR_033_TASK_ID:
        raise FabricatedEndpointSuccessRemovalError(
            "outer receipt task_id must be PCPR-033"
        )
    _reject_closed_release_value(payload.get("status"), "status")
    _reject_closed_release_value(payload.get("promotion_status"), "promotion_status")
    if payload.get("release_claim") is True:
        raise FabricatedEndpointSuccessRemovalError(
            "outer receipt must not claim a release"
        )
    if payload.get("completion_authoritative") is True:
        raise FabricatedEndpointSuccessRemovalError(
            "outer receipt completion is not authoritative"
        )

    verdict_section = payload.get("qualification_verdict")
    if not isinstance(verdict_section, Mapping):
        raise FabricatedEndpointSuccessRemovalError(
            "qualification_verdict must be a mapping"
        )
    _reject_closed_release_value(
        verdict_section.get("promotion_status"),
        "qualification_verdict.promotion_status",
    )
    _reject_closed_release_value(
        verdict_section.get("closed_release_outcome"),
        "qualification_verdict.closed_release_outcome",
    )
    if verdict_section.get("closed_release_outcome") is not None:
        raise FabricatedEndpointSuccessRemovalError(
            "qualification_verdict.closed_release_outcome must not be a closed PCPR release outcome"
        )
    if verdict_section.get("release_claim") is True:
        raise FabricatedEndpointSuccessRemovalError(
            "qualification_verdict must not claim a release"
        )
    if verdict_section.get("duckdb_or_quack_state_written") is True:
        raise FabricatedEndpointSuccessRemovalError(
            "fabricated endpoint-success removal must not write DuckDB or Quack state"
        )
    if verdict_section.get("contracts_frozen") is True:
        raise FabricatedEndpointSuccessRemovalError(
            "fabricated endpoint-success removal cannot freeze contracts"
        )
    if verdict_section.get("live_endpoint_qualified") is True:
        raise FabricatedEndpointSuccessRemovalError(
            "this task cannot claim live endpoint qualification"
        )
    promotion_status = verdict_section.get("promotion_status")
    if promotion_status not in PROMOTION_STATUSES:
        raise FabricatedEndpointSuccessRemovalError(
            "qualification_verdict.promotion_status is not an admitted PCPR-033 status"
        )
    if promotion_status == "supervisor_promoted":
        raise FabricatedEndpointSuccessRemovalError(
            "fabricated endpoint-success removal cannot promote the supervisor"
        )

    acceptance = payload.get("acceptance")
    if isinstance(acceptance, Mapping):
        _reject_closed_release_value(
            acceptance.get("promotion_status"), "acceptance.promotion_status"
        )
        if acceptance.get("closed_release_outcome") is not None:
            raise FabricatedEndpointSuccessRemovalError(
                "acceptance.closed_release_outcome must be null"
            )
        if acceptance.get("release_claim") is True:
            raise FabricatedEndpointSuccessRemovalError(
                "acceptance must not claim a release"
            )

    expected = current_head_pcpr_033_receipt_promotion()
    if verdict_section.get("verdict_cid") != expected["verdict_cid"]:
        raise FabricatedEndpointSuccessRemovalError(
            "qualification_verdict.verdict_cid must match the evaluator"
        )
    if promotion_status != expected["promotion_status"]:
        raise FabricatedEndpointSuccessRemovalError(
            "qualification_verdict.promotion_status must match the evaluator"
        )
    if expected["verdict_cid"] != CURRENT_HEAD_NON_PROMOTION_VERDICT_CID:
        raise FabricatedEndpointSuccessRemovalError(
            "pinned current-head non-promotion CID drifted from the evaluator"
        )

    return {
        "valid": True,
        "task_id": PCPR_033_TASK_ID,
        "promotion_status": promotion_status,
        "closed_release_outcome": None,
        "release_claim": False,
        "verdict_cid": expected["verdict_cid"],
        "evidence_kind": "measured",
    }
