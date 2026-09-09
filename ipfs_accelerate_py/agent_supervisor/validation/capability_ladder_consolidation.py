"""Fail-closed PCPR-034 capability-ladder consolidation.

PCPR-034 makes HardwareCapabilityLadder canonical across remaining
detectors and requires production_authorized for production execution.
Detection, package import, canary, configuration, and advisory
recommendation never grant production. This module is not release
authority: it does not write DuckDB or Quack state, does not qualify
CPU/CUDA, and never emits a closed PCPR release outcome.

Live claims require live evidence. Simulated results stay Simulated.
Missing environments stay typed unavailable.
"""

from __future__ import annotations

import importlib.util
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from ipfs_accelerate_py.assurance.hardware_capability_ladder import (
    CONSOLIDATION_INTERFACE,
    CONSOLIDATION_SCHEMA,
    LADDER_RUNGS,
    PRODUCTION_EXECUTION_CODE,
    admit_production_execution,
    from_canary,
    from_device_visibility,
    from_package_import,
    from_platform_presence,
    production_authorized,
    refused_production_execution,
)
from ..proof.formal_verification_contracts import content_identity
from .canonical_supervisor_contract_freeze import CLOSED_RELEASE_OUTCOMES
from .direct_objective_event_driven_qualification import (
    CURRENT_HEAD_UNAVAILABLE_VERDICT_CID,
    qualify_current_head_without_live_campaign,
)
from .fabricated_endpoint_success_removal import (
    CURRENT_HEAD_NON_PROMOTION_VERDICT_CID as PCPR_033_VERDICT_CID,
)
from .legacy_mock_coordinator_quarantine import (
    discover_accelerate_root,
    discover_portfolio_root,
)
from .source_seal_and_supervisor_baseline import (
    SEALED_GIT_BINARY,
    SEALED_PATH,
    SEALED_PYTHON,
)


CONSOLIDATION_EVALUATOR_INTERFACE: Final = "CapabilityLadderConsolidation@1"
CONSOLIDATION_EVALUATOR_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/capability-ladder-consolidation@1"
)
CONSOLIDATION_VERDICT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "capability-ladder-consolidation-verdict@1"
)

PCPR_034_TASK_ID: Final = "PCPR-034"
PCPR_034_GOAL_ID: Final = "PCPR-G410"
PCPR_033_TASK_ID: Final = "PCPR-033"
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

CURRENT_HEAD_OUTER_COMMIT: Final = "8330efdfbaaa07ffca055b79ae5c764875994e80"
CURRENT_HEAD_OUTER_TREE: Final = "ef43d0251d75562b39e548e22a6282df3da488bd"
CURRENT_HEAD_OUTER_SUBJECT: Final = (
    "Merge commit 'f26b838b25ed4d424d4c24aff65c453f11b5aea1' into "
    "agent/proof-carrying-platform-qualification-and-release-v1"
)
CURRENT_HEAD_ORIGIN_MAIN: Final = "bb8869ed72eb7002434345d9969efee729c4f7f6"
CURRENT_HEAD_ACCELERATOR_COMMIT: Final = (
    "aa263a7f2a6953b38fd200a898f6c8d2aa12fb98"
)
CURRENT_HEAD_ACCELERATOR_TREE: Final = "5e23813edf6420f5e4bf7b47cdd2840d420ae772"
CURRENT_HEAD_ACCELERATOR_ORIGIN_MAIN: Final = (
    "f8c2f633fa6a781b822176fd63e1a229f96b581c"
)
CURRENT_HEAD_DATASETS_COMMIT: Final = "1ab49a5b4c08292963dd177efa4363b5119ebc10"
CURRENT_HEAD_DATASETS_TREE: Final = "30f0117f9b7a4d74e4e48b45ee46200db938c22d"
CURRENT_HEAD_KIT_COMMIT: Final = "1900d8508f49ad7f7dae7bc253d9ed51a7418a70"
CURRENT_HEAD_KIT_TREE: Final = "14e5d8c1786ea1aaf753161dba5665eb44d6d32e"

HERMETIC_CANDIDATE_SUITES: Final[tuple[str, ...]] = (
    "test/api/test_agent_supervisor_capability_ladder_consolidation.py",
)

LADDER_MODULE_RELPATH: Final = (
    "ipfs_accelerate_py/assurance/hardware_capability_ladder.py"
)
DETECTOR_RELPATH: Final = "ipfs_accelerate_py/hf_model_server/hardware/detector.py"
HARDWARE_KIT_RELPATH: Final = "ipfs_accelerate_py/kit/hardware_kit.py"
MCP_HARDWARE_RELPATH: Final = "ipfs_accelerate_py/mcp/tools/hardware.py"
MODEL_LOADER_RELPATH: Final = (
    "ipfs_accelerate_py/hf_model_server/loader/model_loader.py"
)
CLI_RELPATH: Final = "ipfs_accelerate_py/hf_model_server/cli.py"

COMMIT_RE: Final = re.compile(r"^[0-9a-f]{40}$")


class CapabilityLadderConsolidationError(ValueError):
    """Malformed ladder-consolidation evidence or a forbidden authority claim."""


@dataclass(frozen=True)
class ConsolidationProbe:
    """One measured or typed-unavailable ladder-consolidation observation."""

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
class ConsolidationVerdict:
    """Fail-closed PCPR-034 decision. Never a closed release."""

    schema: str
    interface: str
    promotion_status: str
    supervisor_disposition: str
    closed_release_outcome: None
    release_claim: bool
    completion_authoritative: bool
    contracts_frozen: bool
    duckdb_or_quack_state_written: bool
    remaining_detectors_use_canonical_ladder: bool
    production_execution_requires_production_authorized: bool
    detection_implies_production_authorized: bool
    simulated_results_represented_as_live: bool
    live_cuda_qualified: bool
    live_cuda_evidence_kind: str
    live_cpu_qualified: bool
    live_cpu_evidence_kind: str
    this_task_created_competing_authority: bool
    probes: tuple[ConsolidationProbe, ...]
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
            "remaining_detectors_use_canonical_ladder": (
                self.remaining_detectors_use_canonical_ladder
            ),
            "production_execution_requires_production_authorized": (
                self.production_execution_requires_production_authorized
            ),
            "detection_implies_production_authorized": (
                self.detection_implies_production_authorized
            ),
            "simulated_results_represented_as_live": (
                self.simulated_results_represented_as_live
            ),
            "live_cuda_qualified": self.live_cuda_qualified,
            "live_cuda_evidence_kind": self.live_cuda_evidence_kind,
            "live_cpu_qualified": self.live_cpu_qualified,
            "live_cpu_evidence_kind": self.live_cpu_evidence_kind,
            "this_task_created_competing_authority": (
                self.this_task_created_competing_authority
            ),
            "probes": [item.to_mapping() for item in self.probes],
            "blockers": list(self.blockers),
            "verdict_cid": self.verdict_cid,
        }


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CapabilityLadderConsolidationError(f"{name} must be a non-empty string")
    return value.strip()


def _kind(value: Any, name: str) -> str:
    kind = _text(value, name)
    if kind not in EVIDENCE_KINDS:
        raise CapabilityLadderConsolidationError(
            f"{name} is not an admitted evidence kind"
        )
    return kind


def _git_object_id(value: Any, name: str) -> str:
    text = _text(value, name)
    if COMMIT_RE.fullmatch(text) is None:
        raise CapabilityLadderConsolidationError(
            f"{name} must be a 40-character lowercase git object id"
        )
    return text


def _reject_closed_release_value(value: Any, name: str) -> None:
    if isinstance(value, str) and value in CLOSED_RELEASE_OUTCOMES:
        raise CapabilityLadderConsolidationError(
            f"{name} must not be a closed PCPR release outcome"
        )


def _require_ancestor(flag: bool, name: str) -> None:
    if flag is not True:
        raise CapabilityLadderConsolidationError(f"{name} must be true")


def _read_source(root: Path, relpath: str) -> str | None:
    path = root / relpath
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _load_source_module(root: Path, relpath: str, name: str) -> Any | None:
    path = root / relpath
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_contains(source: str | None, *needles: str) -> bool:
    if source is None:
        return False
    return all(needle in source for needle in needles)


def current_head_consolidation_probes(
    *,
    accelerate_root: Path | None = None,
) -> tuple[ConsolidationProbe, ...]:
    """Measured current-tree probes. Missing files stay typed unavailable."""

    root = accelerate_root or discover_accelerate_root()
    probes: list[ConsolidationProbe] = []
    if root is None or not root.is_dir():
        return (
            ConsolidationProbe(
                probe_id="accelerate_source_tree",
                present=None,
                evidence_kind="unavailable",
                live=False,
                simulated_represented_as_live=False,
                reason="Accelerate source tree is not present and is not recorded as empty.",
            ),
        )

    ladder_src = _read_source(root, LADDER_MODULE_RELPATH)
    detector_src = _read_source(root, DETECTOR_RELPATH)
    kit_src = _read_source(root, HARDWARE_KIT_RELPATH)
    mcp_src = _read_source(root, MCP_HARDWARE_RELPATH)
    loader_src = _read_source(root, MODEL_LOADER_RELPATH)
    cli_src = _read_source(root, CLI_RELPATH)

    ladder_ok = _source_contains(
        ladder_src,
        *LADDER_RUNGS,
        "admit_production_execution",
        "from_canary",
        "stamp_consolidation",
        PRODUCTION_EXECUTION_CODE,
    )
    probes.append(
        ConsolidationProbe(
            probe_id="canonical_ladder_module",
            present=None if ladder_src is None else ladder_ok,
            evidence_kind="unavailable" if ladder_src is None else "measured",
            live=False,
            simulated_represented_as_live=False,
            reason=(
                "Canonical ladder rungs and production-execution gate are declared."
                if ladder_ok
                else "Canonical ladder module is missing consolidation APIs."
            ),
            details=MappingProxyType({"relpath": LADDER_MODULE_RELPATH}),
        )
    )

    detector_ok = _source_contains(
        detector_src,
        "from_ladder",
        "get_best_production_hardware",
        'purpose: str = "production"',
        "admit_production_execution",
    )
    probes.append(
        ConsolidationProbe(
            probe_id="detector_uses_canonical_ladder",
            present=None if detector_src is None else detector_ok,
            evidence_kind="unavailable" if detector_src is None else "measured",
            live=False,
            simulated_represented_as_live=False,
            reason=(
                "HardwareDetector constructs capabilities from the canonical ladder "
                "and production selection requires production_authorized."
                if detector_ok
                else "HardwareDetector still selects production hardware from detection."
            ),
            details=MappingProxyType({"relpath": DETECTOR_RELPATH}),
        )
    )

    kit_ok = _source_contains(
        kit_src,
        "stamp_consolidation",
        "from_device_visibility",
        "unavailable_backend",
        "from_canary",
    )
    probes.append(
        ConsolidationProbe(
            probe_id="hardware_kit_remaining_detectors",
            present=None if kit_src is None else kit_ok,
            evidence_kind="unavailable" if kit_src is None else "measured",
            live=False,
            simulated_represented_as_live=False,
            reason=(
                "HardwareKit ROCm, WebGPU, WebNN, and canary tests use the canonical ladder."
                if kit_ok
                else "HardwareKit remaining detectors are not ladder-consolidated."
            ),
            details=MappingProxyType({"relpath": HARDWARE_KIT_RELPATH}),
        )
    )

    mcp_ok = _source_contains(
        mcp_src,
        "stamp_consolidation",
        "unavailable_backend",
        "production_authorized",
    ) and (
        mcp_src is not None
        and 'cuda_info = {"available": False}' not in mcp_src
        and 'webgpu_info = {"available": False}' not in mcp_src
        and 'webnn_info = {"available": False}' not in mcp_src
    )
    probes.append(
        ConsolidationProbe(
            probe_id="mcp_basic_info_no_fabricated_false",
            present=None if mcp_src is None else mcp_ok,
            evidence_kind="unavailable" if mcp_src is None else "measured",
            live=False,
            simulated_represented_as_live=False,
            reason=(
                "MCP basic hardware info no longer fabricates available=False for missing backends."
                if mcp_ok
                else "MCP basic hardware info still fabricates False availability."
            ),
            details=MappingProxyType({"relpath": MCP_HARDWARE_RELPATH}),
        )
    )

    loader_ok = _source_contains(
        loader_src,
        "purpose=\"production\"",
        "admit_production_execution",
        "no production_authorized",
    )
    probes.append(
        ConsolidationProbe(
            probe_id="model_loader_requires_production_authorized",
            present=None if loader_src is None else loader_ok,
            evidence_kind="unavailable" if loader_src is None else "measured",
            live=False,
            simulated_represented_as_live=False,
            reason=(
                "Model load refuses hardware that is not production_authorized."
                if loader_ok
                else "Model loader can still execute production work from detection."
            ),
            details=MappingProxyType({"relpath": MODEL_LOADER_RELPATH}),
        )
    )

    cli_ok = _source_contains(
        cli_src,
        "production_authorized",
        "Detected hardware (not production_authorized)",
    )
    probes.append(
        ConsolidationProbe(
            probe_id="cli_reports_ladder_rungs",
            present=None if cli_src is None else cli_ok,
            evidence_kind="unavailable" if cli_src is None else "measured",
            live=False,
            simulated_represented_as_live=False,
            reason=(
                "CLI hardware listing reports detection separately from production_authorized."
                if cli_ok
                else "CLI hardware listing still treats detection as availability authority."
            ),
            details=MappingProxyType({"relpath": CLI_RELPATH}),
        )
    )

    detector_mod = _load_source_module(
        root, DETECTOR_RELPATH, "pcpr034_hardware_detector"
    )
    if detector_mod is None or not hasattr(detector_mod, "HardwareDetector"):
        probes.append(
            ConsolidationProbe(
                probe_id="detector_not_production_authorized",
                present=None,
                evidence_kind="unavailable",
                live=False,
                simulated_represented_as_live=False,
                reason="HardwareDetector module could not be loaded independently.",
            )
        )
        probes.append(
            ConsolidationProbe(
                probe_id="production_selection_fail_closed",
                present=None,
                evidence_kind="unavailable",
                live=False,
                simulated_represented_as_live=False,
                reason="HardwareSelector could not be loaded independently.",
            )
        )
    else:
        detector = detector_mod.HardwareDetector()
        selector = detector_mod.HardwareSelector(detector)
        cuda_cap = detector.get_capability("cuda")
        cpu_cap = detector.get_capability("cpu")
        detector_authorized = any(
            cap.production_authorized for cap in detector.capabilities.values()
        )
        probes.append(
            ConsolidationProbe(
                probe_id="detector_not_production_authorized",
                present=not detector_authorized
                and cuda_cap is not None
                and cuda_cap.production_authorized is False
                and cpu_cap is not None
                and cpu_cap.production_authorized is False,
                evidence_kind="measured",
                live=False,
                simulated_represented_as_live=False,
                reason=(
                    "HardwareDetector detection flags do not grant production_authorized."
                    if not detector_authorized
                    else "HardwareDetector granted production_authorized."
                ),
                details=MappingProxyType(
                    {
                        "cuda_available": None if cuda_cap is None else cuda_cap.available,
                        "cuda_origin": None if cuda_cap is None else cuda_cap.origin,
                        "cpu_available": None if cpu_cap is None else cpu_cap.available,
                    }
                ),
            )
        )
        production_hw, production_reason = selector.select_hardware(
            {"supported_hardware": ["cuda", "cpu"]},
            purpose="production",
        )
        admission = selector.admit_production_execution(production_hw)
        production_ok = (
            production_hw is None
            and admission.get("admitted") is not True
            and admission.get("code") == PRODUCTION_EXECUTION_CODE
            and admission.get("production_authorized") is False
        )
        probes.append(
            ConsolidationProbe(
                probe_id="production_selection_fail_closed",
                present=production_ok,
                evidence_kind="measured",
                live=False,
                simulated_represented_as_live=False,
                reason=(
                    "Production selection returns no backend because none is production_authorized."
                    if production_ok
                    else "Production selection still chose hardware from detection."
                ),
                details=MappingProxyType(
                    {
                        "hardware": production_hw,
                        "admitted": admission.get("admitted"),
                        "code": admission.get("code"),
                        "reason": production_reason,
                    }
                ),
            )
        )

    kit_mod = _load_source_module(root, HARDWARE_KIT_RELPATH, "pcpr034_hardware_kit")
    if kit_mod is None or not hasattr(kit_mod, "HardwareKit"):
        probes.append(
            ConsolidationProbe(
                probe_id="hardware_kit_typed_unavailable_backends",
                present=None,
                evidence_kind="unavailable",
                live=False,
                simulated_represented_as_live=False,
                reason="HardwareKit module could not be loaded independently.",
            )
        )
        probes.append(
            ConsolidationProbe(
                probe_id="hardware_info_retains_typed_unavailable",
                present=None,
                evidence_kind="unavailable",
                live=False,
                simulated_represented_as_live=False,
                reason="HardwareKit module could not be loaded independently.",
            )
        )
    else:
        kit = kit_mod.HardwareKit()
        cuda_info = kit.detect_cuda()
        rocm_info = kit.detect_rocm()
        metal_info = kit.detect_metal()
        webgpu_info = kit.detect_webgpu()
        webnn_info = kit.detect_webnn()
        kit_ok_runtime = (
            cuda_info.get("production_authorized") is False
            and rocm_info.get("production_authorized") is False
            and metal_info.get("production_authorized") is False
            and webgpu_info.get("available") is None
            and webgpu_info.get("production_authorized") is False
            and webnn_info.get("available") is None
            and webnn_info.get("production_authorized") is False
            and webgpu_info.get("live") is False
            and webnn_info.get("live") is False
        )
        probes.append(
            ConsolidationProbe(
                probe_id="hardware_kit_typed_unavailable_backends",
                present=kit_ok_runtime,
                evidence_kind="measured",
                live=False,
                simulated_represented_as_live=False,
                reason=(
                    "HardwareKit remaining backends stay not production_authorized; "
                    "WebGPU/WebNN absence is typed unavailable."
                    if kit_ok_runtime
                    else "HardwareKit remaining backends still fabricate availability or authorization."
                ),
                details=MappingProxyType(
                    {
                        "cuda_available": cuda_info.get("available"),
                        "rocm_available": rocm_info.get("available"),
                        "metal_available": metal_info.get("available"),
                        "webgpu_available": webgpu_info.get("available"),
                        "webnn_available": webnn_info.get("available"),
                    }
                ),
            )
        )
        hw_info = kit.get_hardware_info(include_detailed=True)
        info_ok = (
            "cuda" in hw_info.accelerators
            and "rocm" in hw_info.accelerators
            and "webgpu" in hw_info.accelerators
            and hw_info.accelerators["cuda"].get("production_authorized") is False
            and hw_info.accelerators["webgpu"].get("available") is None
        )
        probes.append(
            ConsolidationProbe(
                probe_id="hardware_info_retains_typed_unavailable",
                present=info_ok,
                evidence_kind="measured",
                live=False,
                simulated_represented_as_live=False,
                reason=(
                    "Hardware info retains typed-unavailable backends instead of omitting them."
                    if info_ok
                    else "Hardware info still omits unavailable backends or authorizes production."
                ),
            )
        )

    visibility = from_device_visibility("cuda", devices=[{"name": "probe"}])
    imported = from_package_import("openvino", package="openvino", version="fixture")
    platform_report = from_platform_presence("metal", platform_name="Darwin")
    canary = from_canary("cuda", passed=True, extra={"probe": "fixture"})
    visibility_decision = admit_production_execution(visibility)
    canary_decision = admit_production_execution(canary)
    weak_ok = (
        production_authorized(visibility) is False
        and production_authorized(imported) is False
        and production_authorized(platform_report) is False
        and production_authorized(canary) is False
        and visibility_decision.get("admitted") is not True
        and canary_decision.get("admitted") is not True
        and canary.get("canary_passed") is True
        and canary.get("qualified") is False
    )
    probes.append(
        ConsolidationProbe(
            probe_id="weak_evidence_not_authorized",
            present=weak_ok,
            evidence_kind="measured",
            live=False,
            simulated_represented_as_live=False,
            reason=(
                "Device visibility, package import, platform presence, and canary "
                "do not authorize production."
                if weak_ok
                else "Capability ladder still promotes weak evidence to production."
            ),
            details=MappingProxyType(
                {
                    "visibility_authorized": production_authorized(visibility),
                    "import_authorized": production_authorized(imported),
                    "platform_authorized": production_authorized(platform_report),
                    "canary_authorized": production_authorized(canary),
                    "canary_passed": canary.get("canary_passed"),
                }
            ),
        )
    )

    refusal = refused_production_execution("cuda")
    refusal_ok = (
        refusal.get("admitted") is False
        and refusal.get("hardware") is None
        and refusal.get("code") == PRODUCTION_EXECUTION_CODE
        and refusal.get("live") is False
    )
    probes.append(
        ConsolidationProbe(
            probe_id="production_execution_refusal_envelope",
            present=refusal_ok,
            evidence_kind="measured",
            live=False,
            simulated_represented_as_live=False,
            reason=(
                "Production execution refusal is typed unavailable and not live."
                if refusal_ok
                else "Production execution refusal envelope is malformed."
            ),
            details=MappingProxyType(
                {
                    "code": refusal.get("code"),
                    "outcome": refusal.get("outcome"),
                }
            ),
        )
    )

    probes.append(
        ConsolidationProbe(
            probe_id="live_cuda_qualification",
            present=None,
            evidence_kind="unavailable",
            live=False,
            simulated_represented_as_live=False,
            reason=(
                "This task does not qualify live CUDA. Missing GPU evidence stays "
                "typed unavailable and is not recorded as False or passing."
            ),
        )
    )
    probes.append(
        ConsolidationProbe(
            probe_id="live_cpu_qualification",
            present=None,
            evidence_kind="unavailable",
            live=False,
            simulated_represented_as_live=False,
            reason=(
                "This task does not qualify live CPU execution. Declared CPU host "
                "presence is not production_authorized."
            ),
        )
    )
    return tuple(probes)


def qualify_capability_ladder_consolidation(
    *,
    probes: Sequence[ConsolidationProbe],
    duckdb_or_quack_state_written: bool = False,
) -> ConsolidationVerdict:
    """Evaluate PCPR-034. Consolidation is fail-closed; promotion is never a release."""

    if duckdb_or_quack_state_written:
        raise CapabilityLadderConsolidationError(
            "capability-ladder consolidation must not write DuckDB or Quack state"
        )
    if not probes:
        raise CapabilityLadderConsolidationError(
            "capability-ladder-consolidation probes are required"
        )

    normalized: list[ConsolidationProbe] = []
    blockers: list[str] = []
    simulated_as_live = False
    for probe in probes:
        probe_id = _text(probe.probe_id, "probe_id")
        kind = _kind(probe.evidence_kind, f"{probe_id}.evidence_kind")
        if kind == "estimated":
            raise CapabilityLadderConsolidationError(
                f"{probe_id}: estimated values cannot mint ladder-consolidation evidence"
            )
        if kind == "simulated" and probe.present is True:
            raise CapabilityLadderConsolidationError(
                "simulated observations cannot be represented as live presence"
            )
        if kind == "measured_live":
            raise CapabilityLadderConsolidationError(
                f"{probe_id}: this consolidation evaluator cannot carry measured_live evidence"
            )
        if probe.simulated_represented_as_live:
            simulated_as_live = True
            blockers.append(f"{probe_id}:simulated_represented_as_live")
        if probe.live:
            blockers.append(f"{probe_id}:live_claim")
        normalized.append(
            ConsolidationProbe(
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

    remaining = _require(
        "canonical_ladder_module", "canonical_ladder_incomplete"
    )
    remaining = (
        remaining
        and _require("detector_uses_canonical_ladder", "detector_not_consolidated")
        and _require(
            "hardware_kit_remaining_detectors", "hardware_kit_not_consolidated"
        )
        and _require(
            "mcp_basic_info_no_fabricated_false", "mcp_fabricated_false_availability"
        )
        and _require(
            "model_loader_requires_production_authorized",
            "model_loader_uses_detection_as_authority",
        )
        and _require(
            "hardware_kit_typed_unavailable_backends",
            "hardware_kit_fabricates_availability",
        )
        and _require(
            "hardware_info_retains_typed_unavailable",
            "hardware_info_omits_unavailable_backends",
        )
    )
    requires_authorized = _require(
        "production_selection_fail_closed",
        "production_selection_uses_detection",
    )
    _require("detector_not_production_authorized", "detector_authorized_production")
    _require("weak_evidence_not_authorized", "weak_evidence_authorized")
    _require(
        "production_execution_refusal_envelope",
        "production_refusal_envelope_malformed",
    )
    _require("cli_reports_ladder_rungs", "cli_treats_detection_as_authority")

    detection_implies = False
    detector_probe = by_id.get("detector_not_production_authorized")
    if detector_probe is not None and detector_probe.present is not True:
        detection_implies = True

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
        raise CapabilityLadderConsolidationError(
            "internal promotion status is not admitted"
        )
    if promotion_status in CLOSED_RELEASE_OUTCOMES:
        raise CapabilityLadderConsolidationError(
            "capability-ladder consolidation must not mint a closed release outcome"
        )

    payload = {
        "schema": CONSOLIDATION_VERDICT_SCHEMA,
        "interface": CONSOLIDATION_EVALUATOR_INTERFACE,
        "task_id": PCPR_034_TASK_ID,
        "goal_id": PCPR_034_GOAL_ID,
        "promotion_status": promotion_status,
        "supervisor_disposition": "supervisor_non_promoted",
        "closed_release_outcome": None,
        "release_claim": False,
        "completion_authoritative": False,
        "contracts_frozen": False,
        "duckdb_or_quack_state_written": False,
        "remaining_detectors_use_canonical_ladder": remaining,
        "production_execution_requires_production_authorized": requires_authorized,
        "detection_implies_production_authorized": detection_implies,
        "simulated_results_represented_as_live": simulated_as_live,
        "live_cuda_qualified": False,
        "live_cuda_evidence_kind": "unavailable",
        "live_cpu_qualified": False,
        "live_cpu_evidence_kind": "unavailable",
        "this_task_created_competing_authority": False,
        "probes": [item.to_mapping() for item in normalized],
        "blockers": list(dict.fromkeys(blockers)),
    }
    return ConsolidationVerdict(
        schema=CONSOLIDATION_VERDICT_SCHEMA,
        interface=CONSOLIDATION_EVALUATOR_INTERFACE,
        promotion_status=promotion_status,
        supervisor_disposition="supervisor_non_promoted",
        closed_release_outcome=None,
        release_claim=False,
        completion_authoritative=False,
        contracts_frozen=False,
        duckdb_or_quack_state_written=False,
        remaining_detectors_use_canonical_ladder=remaining,
        production_execution_requires_production_authorized=requires_authorized,
        detection_implies_production_authorized=detection_implies,
        simulated_results_represented_as_live=simulated_as_live,
        live_cuda_qualified=False,
        live_cuda_evidence_kind="unavailable",
        live_cpu_qualified=False,
        live_cpu_evidence_kind="unavailable",
        this_task_created_competing_authority=False,
        probes=tuple(normalized),
        blockers=tuple(dict.fromkeys(blockers)),
        verdict_cid=content_identity(payload),
    )


def qualify_current_head_consolidation() -> ConsolidationVerdict:
    """Ordinary current-head PCPR-034 evaluation: honest R&D consolidation."""

    return qualify_capability_ladder_consolidation(
        probes=current_head_consolidation_probes(),
    )


CURRENT_HEAD_NON_PROMOTION_VERDICT_CID: Final = (
    "baguqeerai2yx5egapigykuz5cutne2pzegfbmefskrctkz2jcs7kauot5hsa"
)


def pcpr_034_receipt_promotion(verdict: ConsolidationVerdict) -> dict[str, Any]:
    """Compact outer-receipt promotion fields. Never a closed release."""

    if verdict.closed_release_outcome is not None:
        raise CapabilityLadderConsolidationError(
            "capability-ladder consolidation must not mint a closed release outcome"
        )
    if verdict.release_claim:
        raise CapabilityLadderConsolidationError(
            "capability-ladder consolidation must not claim a PCPR release"
        )
    if verdict.completion_authoritative:
        raise CapabilityLadderConsolidationError(
            "capability-ladder consolidation completion is not authoritative"
        )
    if verdict.duckdb_or_quack_state_written:
        raise CapabilityLadderConsolidationError(
            "capability-ladder consolidation must not write DuckDB or Quack state"
        )
    if verdict.promotion_status in CLOSED_RELEASE_OUTCOMES:
        raise CapabilityLadderConsolidationError(
            "promotion_status must not be a closed release outcome"
        )
    if verdict.promotion_status not in PROMOTION_STATUSES:
        raise CapabilityLadderConsolidationError(
            "promotion_status is not an admitted PCPR-034 status"
        )
    if verdict.promotion_status == "supervisor_promoted":
        raise CapabilityLadderConsolidationError(
            "capability-ladder consolidation cannot promote the supervisor"
        )
    if verdict.live_cuda_qualified or verdict.live_cpu_qualified:
        raise CapabilityLadderConsolidationError(
            "this task cannot claim live CPU or CUDA qualification"
        )
    if verdict.detection_implies_production_authorized:
        raise CapabilityLadderConsolidationError(
            "detection must not imply production_authorized"
        )
    return {
        "schema": CONSOLIDATION_VERDICT_SCHEMA,
        "interface": CONSOLIDATION_EVALUATOR_INTERFACE,
        "verdict_cid": verdict.verdict_cid,
        "promotion_status": verdict.promotion_status,
        "supervisor_disposition": verdict.supervisor_disposition,
        "closed_release_outcome": None,
        "release_claim": False,
        "completion_authoritative": False,
        "contracts_frozen": False,
        "duckdb_or_quack_state_written": False,
        "remaining_detectors_use_canonical_ladder": (
            verdict.remaining_detectors_use_canonical_ladder
        ),
        "production_execution_requires_production_authorized": (
            verdict.production_execution_requires_production_authorized
        ),
        "detection_implies_production_authorized": False,
        "simulated_results_represented_as_live": (
            verdict.simulated_results_represented_as_live
        ),
        "live_cuda_qualified": False,
        "live_cuda_evidence_kind": "unavailable",
        "live_cpu_qualified": False,
        "live_cpu_evidence_kind": "unavailable",
        "this_task_created_competing_authority": (
            verdict.this_task_created_competing_authority
        ),
        "blocker_count": len(verdict.blockers),
        "blockers": list(verdict.blockers),
        "evidence_kind": "measured",
    }


def current_head_pcpr_034_receipt_promotion() -> dict[str, Any]:
    return pcpr_034_receipt_promotion(qualify_current_head_consolidation())


def pcpr_034_receipt_negative_results() -> dict[str, Any]:
    return {
        "simulated_presence_cannot_count_as_live": True,
        "simulated_results_not_represented_as_live": True,
        "estimated_values_rejected": True,
        "hermetic_pass_cannot_qualify_live": True,
        "closed_release_outcome_not_emitted": True,
        "direct_database_bypass_not_used": True,
        "live_cuda_not_claimed": True,
        "live_cpu_not_claimed": True,
        "device_visibility_not_production_authorized": True,
        "package_import_not_production_authorized": True,
        "canary_not_production_authorized": True,
        "detection_not_production_authorized": True,
        "production_execution_requires_production_authorized": True,
        "evidence_kind": "measured",
    }


def current_head_pcpr_034_receipt_sections() -> dict[str, Any]:
    verdict = qualify_current_head_consolidation()
    promotion = pcpr_034_receipt_promotion(verdict)
    qualification = qualify_current_head_without_live_campaign()
    return {
        "qualification_verdict": promotion,
        "qualification_prerequisite": {
            "task_id": PCPR_033_TASK_ID,
            "goal_id": "PCPR-G410",
            "promotion_status": "rnd_non_promoted",
            "removal_verdict_cid": PCPR_033_VERDICT_CID,
            "supervisor_live_qualification_task_id": PCPR_001_TASK_ID,
            "supervisor_live_qualification_promotion_status": qualification.promotion_status,
            "live_qualification_evidence_kind": "unavailable",
            "verdict_cid": CURRENT_HEAD_UNAVAILABLE_VERDICT_CID,
            "closed_release_outcome": None,
            "release_claim": False,
            "completion_authoritative": False,
            "reason": (
                "PCPR-033 removed fabricated endpoint success. This task consolidates "
                "the capability ladder. PCPR-001 live qualification remains "
                "rnd_non_promoted and is not promoted by this receipt."
            ),
            "evidence_kind": "measured",
        },
        "consolidation": {
            "schema": CONSOLIDATION_EVALUATOR_SCHEMA,
            "interface": CONSOLIDATION_EVALUATOR_INTERFACE,
            "ladder_schema": CONSOLIDATION_SCHEMA,
            "ladder_interface": CONSOLIDATION_INTERFACE,
            "remaining_detectors_use_canonical_ladder": (
                verdict.remaining_detectors_use_canonical_ladder
            ),
            "production_execution_requires_production_authorized": (
                verdict.production_execution_requires_production_authorized
            ),
            "detection_implies_production_authorized": False,
            "probes": [item.to_mapping() for item in verdict.probes],
            "evidence_kind": "measured",
        },
        "negative_results": pcpr_034_receipt_negative_results(),
        "verdict_cid": verdict.verdict_cid,
        "promotion_status": promotion["promotion_status"],
        "closed_release_outcome": None,
        "release_claim": False,
        "contracts_frozen": False,
    }


def pcpr_034_current_tree_binding(
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
        raise CapabilityLadderConsolidationError(
            "accelerator_pre_change_commit must equal accelerator_gitlink"
        )
    datasets = _git_object_id(datasets_commit, "datasets_commit")
    datasets_tree_id = _git_object_id(datasets_tree, "datasets_tree")
    datasets_link = _git_object_id(datasets_gitlink, "datasets_gitlink")
    if datasets != datasets_link:
        raise CapabilityLadderConsolidationError(
            "datasets_commit must equal datasets_gitlink"
        )
    kit = _git_object_id(kit_commit, "kit_commit")
    kit_tree_id = _git_object_id(kit_tree, "kit_tree")
    kit_link = _git_object_id(kit_gitlink, "kit_gitlink")
    if kit != kit_link:
        raise CapabilityLadderConsolidationError(
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


def current_head_pcpr_034_current_tree_binding() -> dict[str, Any]:
    return pcpr_034_current_tree_binding(
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


def validate_pcpr_034_outer_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed if an outer PCPR-034 receipt claims a closed release."""

    if not isinstance(payload, Mapping):
        raise CapabilityLadderConsolidationError("outer receipt must be a mapping")
    if payload.get("task_id") != PCPR_034_TASK_ID:
        raise CapabilityLadderConsolidationError(
            "outer receipt task_id must be PCPR-034"
        )
    _reject_closed_release_value(payload.get("status"), "status")
    _reject_closed_release_value(payload.get("promotion_status"), "promotion_status")
    if payload.get("release_claim") is True:
        raise CapabilityLadderConsolidationError(
            "outer receipt must not claim a release"
        )
    if payload.get("completion_authoritative") is True:
        raise CapabilityLadderConsolidationError(
            "outer receipt completion is not authoritative"
        )

    verdict_section = payload.get("qualification_verdict")
    if not isinstance(verdict_section, Mapping):
        raise CapabilityLadderConsolidationError(
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
        raise CapabilityLadderConsolidationError(
            "qualification_verdict.closed_release_outcome must not be a closed PCPR release outcome"
        )
    if verdict_section.get("release_claim") is True:
        raise CapabilityLadderConsolidationError(
            "qualification_verdict must not claim a release"
        )
    if verdict_section.get("duckdb_or_quack_state_written") is True:
        raise CapabilityLadderConsolidationError(
            "capability-ladder consolidation must not write DuckDB or Quack state"
        )
    if verdict_section.get("contracts_frozen") is True:
        raise CapabilityLadderConsolidationError(
            "capability-ladder consolidation cannot freeze contracts"
        )
    if verdict_section.get("live_cuda_qualified") is True:
        raise CapabilityLadderConsolidationError(
            "this task cannot claim live CUDA qualification"
        )
    if verdict_section.get("live_cpu_qualified") is True:
        raise CapabilityLadderConsolidationError(
            "this task cannot claim live CPU qualification"
        )
    if verdict_section.get("detection_implies_production_authorized") is True:
        raise CapabilityLadderConsolidationError(
            "detection must not imply production_authorized"
        )
    promotion_status = verdict_section.get("promotion_status")
    if promotion_status not in PROMOTION_STATUSES:
        raise CapabilityLadderConsolidationError(
            "qualification_verdict.promotion_status is not an admitted PCPR-034 status"
        )
    if promotion_status == "supervisor_promoted":
        raise CapabilityLadderConsolidationError(
            "capability-ladder consolidation cannot promote the supervisor"
        )

    acceptance = payload.get("acceptance")
    if isinstance(acceptance, Mapping):
        _reject_closed_release_value(
            acceptance.get("promotion_status"), "acceptance.promotion_status"
        )
        if acceptance.get("closed_release_outcome") is not None:
            raise CapabilityLadderConsolidationError(
                "acceptance.closed_release_outcome must be null"
            )
        if acceptance.get("release_claim") is True:
            raise CapabilityLadderConsolidationError(
                "acceptance must not claim a release"
            )

    expected = current_head_pcpr_034_receipt_promotion()
    if verdict_section.get("verdict_cid") != expected["verdict_cid"]:
        raise CapabilityLadderConsolidationError(
            "qualification_verdict.verdict_cid must match the evaluator"
        )
    if promotion_status != expected["promotion_status"]:
        raise CapabilityLadderConsolidationError(
            "qualification_verdict.promotion_status must match the evaluator"
        )
    if expected["verdict_cid"] != CURRENT_HEAD_NON_PROMOTION_VERDICT_CID:
        raise CapabilityLadderConsolidationError(
            "pinned current-head non-promotion CID drifted from the evaluator"
        )

    return {
        "valid": True,
        "task_id": PCPR_034_TASK_ID,
        "promotion_status": promotion_status,
        "closed_release_outcome": None,
        "release_claim": False,
        "verdict_cid": expected["verdict_cid"],
        "evidence_kind": "measured",
    }
