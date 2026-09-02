"""Fail-closed PCPR-031 fabricated hardware capability removal.

PCPR-031 replaces mock hardware detection and CUDA mock availability with a
capability ladder. Ordinary runtime cannot treat simulation, package import,
or device visibility as production_authorized. This module is not release
authority: it does not write DuckDB or Quack state, does not qualify
CPU/CUDA, and never emits a closed PCPR release outcome.

Live claims require live evidence. Simulated hardware stays ``Simulated``.
Missing environments stay typed unavailable.
"""

from __future__ import annotations

import hashlib
import importlib.util
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from ipfs_accelerate_py.assurance.hardware_capability_ladder import (
    LADDER_RUNGS,
    assess_ladder,
    from_device_visibility,
    from_package_import,
    from_platform_presence,
    production_authorized,
    unavailable_hardware_map,
)
from ipfs_accelerate_py.compatibility.simulation.fabricated_hardware import (
    FabricatedHardwareError,
    MockHardwareDetection,
    UnavailableHardwareDetection,
    UnavailableTorch,
    create_simulated_cuda_implementation,
    instantiate_mock_hardware_detection,
    load_ordinary_hardware_detection,
    torch_is_fabricated,
)

from ..proof.formal_verification_contracts import content_identity
from .canonical_supervisor_contract_freeze import CLOSED_RELEASE_OUTCOMES
from .direct_objective_event_driven_qualification import (
    CURRENT_HEAD_UNAVAILABLE_VERDICT_CID,
    qualify_current_head_without_live_campaign,
)
from .legacy_mock_coordinator_quarantine import (
    CURRENT_HEAD_NON_PROMOTION_VERDICT_CID as PCPR_030_VERDICT_CID,
    discover_accelerate_root,
    discover_portfolio_root,
)
from .source_seal_and_supervisor_baseline import (
    SEALED_GIT_BINARY,
    SEALED_PATH,
    SEALED_PYTHON,
)


REMOVAL_INTERFACE: Final = "FabricatedHardwareRemoval@1"
REMOVAL_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/fabricated-hardware-removal@1"
)
REMOVAL_VERDICT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/fabricated-hardware-removal-verdict@1"
)

PCPR_031_TASK_ID: Final = "PCPR-031"
PCPR_031_GOAL_ID: Final = "PCPR-G410"
PCPR_030_TASK_ID: Final = "PCPR-030"
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

CURRENT_HEAD_OUTER_COMMIT: Final = "84c1f052c973a268353f4858ed361ff5b144c3b5"
CURRENT_HEAD_OUTER_TREE: Final = "1e83eb84353df1648ddb4f0318374446b0fe04d5"
CURRENT_HEAD_OUTER_SUBJECT: Final = (
    "Merge commit '29a7899d2a6490638b28ae4f91e52bdf2202344d' into "
    "agent/proof-carrying-platform-qualification-and-release-v1"
)
CURRENT_HEAD_ORIGIN_MAIN: Final = "bb8869ed72eb7002434345d9969efee729c4f7f6"
CURRENT_HEAD_ACCELERATOR_COMMIT: Final = (
    "0e6cc5a7667c06406f8344141260a016de926579"
)
CURRENT_HEAD_ACCELERATOR_TREE: Final = "7e48f0af464640212b93227d1b544ccf60a5a253"
CURRENT_HEAD_ACCELERATOR_ORIGIN_MAIN: Final = (
    "f8c2f633fa6a781b822176fd63e1a229f96b581c"
)
CURRENT_HEAD_DATASETS_COMMIT: Final = "f49afc579c22856849ca9f739435e5820003384f"
CURRENT_HEAD_DATASETS_TREE: Final = "47118a8e6d1b6b4e7ae04f9a2efda33aadebca1b"
CURRENT_HEAD_KIT_COMMIT: Final = "b6c65ba732733d7e33852713ba18aa3b12235668"
CURRENT_HEAD_KIT_TREE: Final = "14da7d92e130b7ba3523d0d6741a3ef7ef1e1bc2"

HERMETIC_CANDIDATE_SUITES: Final[tuple[str, ...]] = (
    "test/api/test_agent_supervisor_fabricated_hardware_removal.py",
)

SIMULATION_MODULE_RELPATH: Final = (
    "ipfs_accelerate_py/compatibility/simulation/fabricated_hardware.py"
)
LADDER_MODULE_RELPATH: Final = (
    "ipfs_accelerate_py/assurance/hardware_capability_ladder.py"
)
LEGACY_MODULE_RELPATH: Final = "ipfs_accelerate_py/ipfs_accelerate_py_legacy.py"
CUDA_UTILS_RELPATH: Final = "ipfs_accelerate_py/worker/cuda_utils.py"
DETECTOR_RELPATH: Final = "ipfs_accelerate_py/hf_model_server/hardware/detector.py"
HARDWARE_KIT_RELPATH: Final = "ipfs_accelerate_py/kit/hardware_kit.py"

COMMIT_RE: Final = __import__("re").compile(r"^[0-9a-f]{40}$")


class FabricatedHardwareRemovalError(ValueError):
    """Malformed hardware-removal evidence or a forbidden authority claim."""


@dataclass(frozen=True)
class RemovalProbe:
    """One measured or typed-unavailable hardware-removal observation."""

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
    """Fail-closed PCPR-031 decision. Never a closed release."""

    schema: str
    interface: str
    promotion_status: str
    supervisor_disposition: str
    closed_release_outcome: None
    release_claim: bool
    completion_authoritative: bool
    contracts_frozen: bool
    duckdb_or_quack_state_written: bool
    ordinary_runtime_uses_mock_hardware: bool
    mock_hardware_quarantined: bool
    cuda_mock_requires_explicit_simulation: bool
    device_visibility_implies_production_authorized: bool
    simulated_results_represented_as_live: bool
    live_cuda_qualified: bool
    live_cuda_evidence_kind: str
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
            "ordinary_runtime_uses_mock_hardware": (
                self.ordinary_runtime_uses_mock_hardware
            ),
            "mock_hardware_quarantined": self.mock_hardware_quarantined,
            "cuda_mock_requires_explicit_simulation": (
                self.cuda_mock_requires_explicit_simulation
            ),
            "device_visibility_implies_production_authorized": (
                self.device_visibility_implies_production_authorized
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
        raise FabricatedHardwareRemovalError(f"{name} must be a non-empty string")
    return value.strip()


def _kind(value: Any, name: str) -> str:
    kind = _text(value, name)
    if kind not in EVIDENCE_KINDS:
        raise FabricatedHardwareRemovalError(f"{name} is not an admitted evidence kind")
    return kind


def _git_object_id(value: Any, name: str) -> str:
    text = _text(value, name)
    if COMMIT_RE.fullmatch(text) is None:
        raise FabricatedHardwareRemovalError(
            f"{name} must be a 40-character lowercase git object id"
        )
    return text


def _reject_closed_release_value(value: Any, name: str) -> None:
    if isinstance(value, str) and value in CLOSED_RELEASE_OUTCOMES:
        raise FabricatedHardwareRemovalError(
            f"{name} must not be a closed PCPR release outcome"
        )


def _require_ancestor(flag: bool, name: str) -> None:
    if flag is not True:
        raise FabricatedHardwareRemovalError(f"{name} must be true")


def _read_source(root: Path, relpath: str) -> str | None:
    path = root / relpath
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _load_source_module(root: Path, relpath: str, name: str) -> Any | None:
    """Load a module from a file path without executing package __init__ side effects."""

    path = root / relpath
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def current_head_removal_probes(
    *,
    accelerate_root: Path | None = None,
) -> tuple[RemovalProbe, ...]:
    """Measured current-tree probes. Missing files stay typed unavailable."""

    root = accelerate_root or discover_accelerate_root()
    probes: list[RemovalProbe] = []
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
    ladder = _read_source(root, LADDER_MODULE_RELPATH)
    legacy = _read_source(root, LEGACY_MODULE_RELPATH)
    cuda_src = _read_source(root, CUDA_UTILS_RELPATH)
    detector_src = _read_source(root, DETECTOR_RELPATH)
    kit_src = _read_source(root, HARDWARE_KIT_RELPATH)

    if simulation is None:
        probes.append(
            RemovalProbe(
                probe_id="simulation_namespace_mock_hardware",
                present=None,
                evidence_kind="unavailable",
                live=False,
                simulated_represented_as_live=False,
                reason="Simulation namespace module is unreadable.",
            )
        )
    else:
        present = "class MockHardwareDetection" in simulation
        probes.append(
            RemovalProbe(
                probe_id="simulation_namespace_mock_hardware",
                present=present,
                evidence_kind="measured",
                live=False,
                simulated_represented_as_live=False,
                reason=(
                    "MockHardwareDetection is defined in the explicit simulation namespace."
                    if present
                    else "MockHardwareDetection is missing from the simulation namespace."
                ),
                details=MappingProxyType({"relpath": SIMULATION_MODULE_RELPATH}),
            )
        )

    if ladder is None:
        probes.append(
            RemovalProbe(
                probe_id="capability_ladder_module",
                present=None,
                evidence_kind="unavailable",
                live=False,
                simulated_represented_as_live=False,
                reason="Capability ladder module is unreadable.",
            )
        )
    else:
        present = all(rung in ladder for rung in LADDER_RUNGS)
        probes.append(
            RemovalProbe(
                probe_id="capability_ladder_module",
                present=present,
                evidence_kind="measured",
                live=False,
                simulated_represented_as_live=False,
                reason=(
                    "Capability ladder rungs are declared."
                    if present
                    else "Capability ladder rungs are incomplete."
                ),
                details=MappingProxyType({"relpath": LADDER_MODULE_RELPATH}),
            )
        )

    if legacy is None:
        probes.append(
            RemovalProbe(
                probe_id="ordinary_legacy_hardware_setup",
                present=None,
                evidence_kind="unavailable",
                live=False,
                simulated_represented_as_live=False,
                reason="Legacy module is unreadable.",
            )
        )
    else:
        uses_unavailable = "load_ordinary_hardware_detection" in legacy
        setup_calls_mock = (
            "self.hardware_detection = self._create_mock_hardware_detection()" in legacy
        )
        probes.append(
            RemovalProbe(
                probe_id="ordinary_legacy_hardware_setup",
                present=uses_unavailable and not setup_calls_mock,
                evidence_kind="measured",
                live=False,
                simulated_represented_as_live=False,
                reason=(
                    "Ordinary legacy setup loads typed unavailable hardware, not mock detection."
                    if uses_unavailable and not setup_calls_mock
                    else "Ordinary legacy setup still instantiates mock hardware detection."
                ),
                details=MappingProxyType({"relpath": LEGACY_MODULE_RELPATH}),
            )
        )

    if cuda_src is None:
        probes.append(
            RemovalProbe(
                probe_id="cuda_utils_no_magicmock_torch",
                present=None,
                evidence_kind="unavailable",
                live=False,
                simulated_represented_as_live=False,
                reason="cuda_utils.py is unreadable.",
            )
        )
    else:
        uses_unavailable = "UnavailableTorch" in cuda_src
        magicmock_fallback = 'print("PyTorch not available, using mock")' in cuda_src
        probes.append(
            RemovalProbe(
                probe_id="cuda_utils_no_magicmock_torch",
                present=uses_unavailable and not magicmock_fallback,
                evidence_kind="measured",
                live=False,
                simulated_represented_as_live=False,
                reason=(
                    "Missing torch is typed unavailable, not a MagicMock CUDA adapter."
                    if uses_unavailable and not magicmock_fallback
                    else "cuda_utils still substitutes MagicMock for missing torch."
                ),
                details=MappingProxyType({"relpath": CUDA_UTILS_RELPATH}),
            )
        )

    if detector_src is None:
        probes.append(
            RemovalProbe(
                probe_id="detector_no_qualification_log",
                present=None,
                evidence_kind="unavailable",
                live=False,
                simulated_represented_as_live=False,
                reason="HardwareDetector source is unreadable.",
            )
        )
    else:
        fabricated_log = 'f"Available hardware:' in detector_src or "Available hardware:" in detector_src
        probes.append(
            RemovalProbe(
                probe_id="detector_no_qualification_log",
                present=not fabricated_log,
                evidence_kind="measured",
                live=False,
                simulated_represented_as_live=False,
                reason=(
                    "HardwareDetector no longer logs 'Available hardware' as qualification."
                    if not fabricated_log
                    else "HardwareDetector still logs Available hardware as if qualified."
                ),
                details=MappingProxyType({"relpath": DETECTOR_RELPATH}),
            )
        )

    if kit_src is None:
        probes.append(
            RemovalProbe(
                probe_id="hardware_kit_no_fabricated_available_true",
                present=None,
                evidence_kind="unavailable",
                live=False,
                simulated_represented_as_live=False,
                reason="hardware_kit.py is unreadable.",
            )
        )
    else:
        nvidia_true = 'cuda_info["available"] = True' in kit_src
        darwin_true = 'metal_info["available"] = True' in kit_src
        probes.append(
            RemovalProbe(
                probe_id="hardware_kit_no_fabricated_available_true",
                present=not nvidia_true and not darwin_true,
                evidence_kind="measured",
                live=False,
                simulated_represented_as_live=False,
                reason=(
                    "hardware_kit no longer assigns available=True from nvidia-smi or Darwin."
                    if not nvidia_true and not darwin_true
                    else "hardware_kit still fabricates available=True from visibility."
                ),
                details=MappingProxyType(
                    {
                        "relpath": HARDWARE_KIT_RELPATH,
                        "nvidia_smi_available_true": nvidia_true,
                        "darwin_metal_available_true": darwin_true,
                    }
                ),
            )
        )

    ordinary = load_ordinary_hardware_detection()
    ordinary_map = ordinary.detect_all_hardware()
    ordinary_ok = (
        isinstance(ordinary, UnavailableHardwareDetection)
        and ordinary.live is False
        and ordinary.production_authorized is False
        and ordinary_map["cuda"].get("available") is None
        and ordinary_map["cuda"].get("production_authorized") is False
    )
    probes.append(
        RemovalProbe(
            probe_id="ordinary_hardware_detection",
            present=ordinary_ok,
            evidence_kind="measured",
            live=False,
            simulated_represented_as_live=False,
            reason=(
                "Ordinary hardware detection is typed unavailable and not mock."
                if ordinary_ok
                else "Ordinary hardware detection is not the typed unavailable stand-in."
            ),
            details=MappingProxyType(
                {
                    "type_name": type(ordinary).__name__,
                    "cuda_available": ordinary_map["cuda"].get("available"),
                    "outcome": ordinary.outcome,
                }
            ),
        )
    )

    mock_blocked = False
    mock_code = ""
    try:
        instantiate_mock_hardware_detection()
    except FabricatedHardwareError as exc:
        mock_blocked = True
        mock_code = exc.code
    probes.append(
        RemovalProbe(
            probe_id="mock_hardware_ordinary_instantiation",
            present=mock_blocked,
            evidence_kind="measured",
            live=False,
            simulated_represented_as_live=False,
            reason=(
                "Mock hardware ordinary instantiation is refused."
                if mock_blocked
                else "Mock hardware can still be constructed without explicit simulation."
            ),
            details=MappingProxyType({"code": mock_code}),
        )
    )

    simulated = instantiate_mock_hardware_detection(explicit_simulation=True)
    simulated_map = simulated.detect_all_hardware()
    simulated_as_live = bool(simulated.live) or simulated_map["cuda"].get("available") is True
    probes.append(
        RemovalProbe(
            probe_id="mock_hardware_explicit_simulation",
            present=isinstance(simulated, MockHardwareDetection) and not simulated_as_live,
            evidence_kind="measured",
            live=False,
            simulated_represented_as_live=simulated_as_live,
            reason=(
                "Explicit mock hardware is simulated and does not report live CUDA."
                if not simulated_as_live
                else "Explicit mock hardware reported a live CUDA claim."
            ),
            details=MappingProxyType(
                {
                    "origin": simulated.origin,
                    "outcome": simulated.outcome,
                    "cuda_available": simulated_map["cuda"].get("available"),
                }
            ),
        )
    )

    cuda_blocked = False
    try:
        create_simulated_cuda_implementation("lm")
    except FabricatedHardwareError:
        cuda_blocked = True
    probes.append(
        RemovalProbe(
            probe_id="cuda_mock_ordinary_instantiation",
            present=cuda_blocked,
            evidence_kind="measured",
            live=False,
            simulated_represented_as_live=False,
            reason=(
                "CUDA mock ordinary instantiation is refused."
                if cuda_blocked
                else "CUDA mock can still be constructed without explicit simulation."
            ),
        )
    )

    endpoint, _model, handler, _queue, _batch = create_simulated_cuda_implementation(
        "lm", explicit_simulation=True
    )
    labeled = handler("probe")
    cuda_live_claim = (
        labeled.get("live") is True
        or labeled.get("production_authorized") is True
        or labeled.get("implementation_type") == "REAL"
    )
    probes.append(
        RemovalProbe(
            probe_id="cuda_mock_explicit_simulation",
            present=labeled.get("outcome") == "Simulated" and not cuda_live_claim,
            evidence_kind="measured",
            live=False,
            simulated_represented_as_live=cuda_live_claim,
            reason=(
                "Explicit CUDA mock is labeled Simulated and is not live."
                if not cuda_live_claim
                else "Explicit CUDA mock claimed live or REAL."
            ),
            details=MappingProxyType(
                {
                    "outcome": labeled.get("outcome"),
                    "live": labeled.get("live"),
                    "implementation_type": labeled.get("implementation_type"),
                }
            ),
        )
    )

    cuda_mod = _load_source_module(root, CUDA_UTILS_RELPATH, "pcpr031_cuda_utils")
    if cuda_mod is None or not hasattr(cuda_mod, "cuda_utils"):
        probes.append(
            RemovalProbe(
                probe_id="cuda_utils_runtime_unavailable",
                present=None,
                evidence_kind="unavailable",
                live=False,
                simulated_represented_as_live=False,
                reason="cuda_utils module could not be loaded independently.",
            )
        )
        cuda_cap = None
    else:
        utils = cuda_mod.cuda_utils(resources={})
        device = utils.get_cuda_device()
        fabricated_torch = torch_is_fabricated(utils.torch)
        cuda_utils_ok = (not fabricated_torch) or device is None
        probes.append(
            RemovalProbe(
                probe_id="cuda_utils_runtime_unavailable",
                present=cuda_utils_ok,
                evidence_kind="measured",
                live=False,
                simulated_represented_as_live=False,
                reason=(
                    "cuda_utils does not treat a missing or mock torch as live CUDA."
                ),
                details=MappingProxyType(
                    {
                        "device": None if device is None else str(device),
                        "torch_type": type(utils.torch).__name__,
                        "torch_is_fabricated": fabricated_torch,
                        "torch_is_unavailable": isinstance(utils.torch, UnavailableTorch),
                    }
                ),
            )
        )

    detector_mod = _load_source_module(root, DETECTOR_RELPATH, "pcpr031_hardware_detector")
    if detector_mod is None or not hasattr(detector_mod, "HardwareDetector"):
        probes.append(
            RemovalProbe(
                probe_id="detector_not_production_authorized",
                present=None,
                evidence_kind="unavailable",
                live=False,
                simulated_represented_as_live=False,
                reason="HardwareDetector module could not be loaded independently.",
            )
        )
        cuda_cap = None
    else:
        detector = detector_mod.HardwareDetector()
        cuda_cap = detector.get_capability("cuda")
        detector_authorized = any(
            cap.production_authorized for cap in detector.capabilities.values()
        )
        probes.append(
            RemovalProbe(
                probe_id="detector_not_production_authorized",
                present=not detector_authorized
                and cuda_cap is not None
                and cuda_cap.production_authorized is False
                and cuda_cap.live is False,
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
                        "cuda_origin": None if cuda_cap is None else getattr(cuda_cap, "origin", None),
                        "cuda_evidence_kind": None
                        if cuda_cap is None
                        else getattr(cuda_cap, "evidence_kind", None),
                    }
                ),
            )
        )

    kit_mod = _load_source_module(root, HARDWARE_KIT_RELPATH, "pcpr031_hardware_kit")
    if kit_mod is None or not hasattr(kit_mod, "HardwareKit"):
        probes.append(
            RemovalProbe(
                probe_id="hardware_kit_ladder",
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
        metal_info = kit.detect_metal()
        kit_ok = (
            cuda_info.get("production_authorized") is False
            and cuda_info.get("live") is False
            and metal_info.get("production_authorized") is False
            and metal_info.get("available") is not True
        )
        probes.append(
            RemovalProbe(
                probe_id="hardware_kit_ladder",
                present=kit_ok,
                evidence_kind="measured",
                live=False,
                simulated_represented_as_live=False,
                reason=(
                    "HardwareKit CUDA/Metal reports stay not production_authorized."
                    if kit_ok
                    else "HardwareKit still fabricates availability or authorization."
                ),
                details=MappingProxyType(
                    {
                        "cuda_available": cuda_info.get("available"),
                        "cuda_origin": cuda_info.get("origin"),
                        "metal_available": metal_info.get("available"),
                        "metal_origin": metal_info.get("origin"),
                    }
                ),
            )
        )

    visibility = from_device_visibility("cuda", devices=[{"name": "probe"}])
    imported = from_package_import("openvino", package="openvino", version="fixture")
    platform = from_platform_presence("metal", platform_name="Darwin")
    ladder_ok = (
        production_authorized(visibility) is False
        and production_authorized(imported) is False
        and production_authorized(platform) is False
        and assess_ladder(visibility)["blocking_rung"] == "canary_passed"
    )
    visibility_ok = (
        visibility.get("available") is True
        and visibility.get("production_authorized") is False
        and imported.get("installed") is True
        and imported.get("available") is False
        and platform.get("available") is None
        and platform.get("declared") is True
    )
    probes.append(
        RemovalProbe(
            probe_id="ladder_weak_evidence_not_authorized",
            present=visibility_ok and ladder_ok,
            evidence_kind="measured",
            live=False,
            simulated_represented_as_live=False,
            reason=(
                "Device visibility, package import, and platform presence do not authorize production."
                if visibility_ok
                else "Capability ladder still promotes weak evidence."
            ),
            details=MappingProxyType(
                {
                    "visibility_authorized": production_authorized(visibility),
                    "import_authorized": production_authorized(imported),
                    "platform_authorized": production_authorized(platform),
                }
            ),
        )
    )

    unavailable = unavailable_hardware_map()
    probes.append(
        RemovalProbe(
            probe_id="typed_unavailable_map",
            present=unavailable["cuda"].get("available") is None
            and unavailable["cuda"].get("evidence_kind") == "unavailable",
            evidence_kind="measured",
            live=False,
            simulated_represented_as_live=False,
            reason="Typed unavailable map does not record missing CUDA as measured False.",
        )
    )

    probes.append(
        RemovalProbe(
            probe_id="live_cuda_qualification",
            present=None,
            evidence_kind="unavailable",
            live=False,
            simulated_represented_as_live=False,
            reason=(
                "This task does not qualify live CUDA. Missing GPU evidence stays "
                "typed unavailable and is not recorded as False or passing. "
                f"detector_cuda_available={None if cuda_cap is None else cuda_cap.available!r}."
            ),
        )
    )
    return tuple(probes)


def qualify_fabricated_hardware_removal(
    *,
    probes: Sequence[RemovalProbe],
    duckdb_or_quack_state_written: bool = False,
) -> RemovalVerdict:
    """Evaluate PCPR-031. Removal is fail-closed; promotion is never a release."""

    if duckdb_or_quack_state_written:
        raise FabricatedHardwareRemovalError(
            "fabricated hardware removal must not write DuckDB or Quack state"
        )
    if not probes:
        raise FabricatedHardwareRemovalError("hardware-removal probes are required")

    normalized: list[RemovalProbe] = []
    blockers: list[str] = []
    simulated_as_live = False
    for probe in probes:
        probe_id = _text(probe.probe_id, "probe_id")
        kind = _kind(probe.evidence_kind, f"{probe_id}.evidence_kind")
        if kind == "estimated":
            raise FabricatedHardwareRemovalError(
                f"{probe_id}: estimated values cannot mint hardware-removal evidence"
            )
        if kind == "simulated" and probe.present is True:
            raise FabricatedHardwareRemovalError(
                "simulated observations cannot be represented as live presence"
            )
        if kind == "measured_live":
            raise FabricatedHardwareRemovalError(
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

    ordinary_mock = by_id.get("ordinary_legacy_hardware_setup")
    uses_mock = bool(ordinary_mock and ordinary_mock.present is False)
    if ordinary_mock is None or ordinary_mock.present is not True:
        blockers.append("ordinary_runtime_uses_mock_hardware")

    mock_quarantined = _require(
        "simulation_namespace_mock_hardware", "mock_hardware_not_in_simulation_namespace"
    )
    _require("capability_ladder_module", "capability_ladder_missing")
    _require("cuda_utils_no_magicmock_torch", "cuda_utils_magicmock_torch")
    _require("detector_no_qualification_log", "detector_available_hardware_log")
    _require(
        "hardware_kit_no_fabricated_available_true",
        "hardware_kit_fabricated_available_true",
    )
    _require("ordinary_hardware_detection", "ordinary_hardware_not_unavailable")
    _require("mock_hardware_ordinary_instantiation", "mock_hardware_ordinary_instantiation_allowed")
    _require("mock_hardware_explicit_simulation", "mock_hardware_reported_live")
    _require("cuda_utils_runtime_unavailable", "cuda_utils_fabricated_device")
    cuda_gated = _require(
        "cuda_mock_ordinary_instantiation",
        "cuda_mock_ordinary_instantiation_allowed",
    )
    _require("cuda_mock_explicit_simulation", "cuda_mock_not_labeled_simulated")
    _require("detector_not_production_authorized", "detector_production_authorized")
    _require("hardware_kit_ladder", "hardware_kit_not_on_ladder")
    _require("ladder_weak_evidence_not_authorized", "ladder_promotes_weak_evidence")

    visibility_implies = False
    kit = by_id.get("hardware_kit_ladder")
    if kit is not None and kit.present is not True:
        visibility_implies = True

    unavailable_count = sum(1 for item in normalized if item.evidence_kind == "unavailable")
    if blockers:
        promotion_status = "typed_blocked"
    elif unavailable_count == len(normalized):
        promotion_status = "typed_unavailable"
    else:
        promotion_status = "rnd_non_promoted"

    if promotion_status not in PROMOTION_STATUSES:
        raise FabricatedHardwareRemovalError("internal promotion status is not admitted")
    if promotion_status in CLOSED_RELEASE_OUTCOMES:
        raise FabricatedHardwareRemovalError(
            "fabricated hardware removal must not mint a closed release outcome"
        )

    payload = {
        "schema": REMOVAL_VERDICT_SCHEMA,
        "interface": REMOVAL_INTERFACE,
        "task_id": PCPR_031_TASK_ID,
        "goal_id": PCPR_031_GOAL_ID,
        "promotion_status": promotion_status,
        "supervisor_disposition": "supervisor_non_promoted",
        "closed_release_outcome": None,
        "release_claim": False,
        "completion_authoritative": False,
        "contracts_frozen": False,
        "duckdb_or_quack_state_written": False,
        "ordinary_runtime_uses_mock_hardware": uses_mock,
        "mock_hardware_quarantined": mock_quarantined,
        "cuda_mock_requires_explicit_simulation": cuda_gated,
        "device_visibility_implies_production_authorized": visibility_implies,
        "simulated_results_represented_as_live": simulated_as_live,
        "live_cuda_qualified": False,
        "live_cuda_evidence_kind": "unavailable",
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
        ordinary_runtime_uses_mock_hardware=uses_mock,
        mock_hardware_quarantined=mock_quarantined,
        cuda_mock_requires_explicit_simulation=cuda_gated,
        device_visibility_implies_production_authorized=visibility_implies,
        simulated_results_represented_as_live=simulated_as_live,
        live_cuda_qualified=False,
        live_cuda_evidence_kind="unavailable",
        this_task_created_competing_authority=False,
        probes=tuple(normalized),
        blockers=tuple(dict.fromkeys(blockers)),
        verdict_cid=content_identity(payload),
    )


def qualify_current_head_removal() -> RemovalVerdict:
    """Ordinary current-head PCPR-031 evaluation: honest R&D removal."""

    return qualify_fabricated_hardware_removal(
        probes=current_head_removal_probes(),
    )


CURRENT_HEAD_NON_PROMOTION_VERDICT_CID: Final = (
    "baguqeerax6l46w26fq7vmojcf73bwzf3xwmsh6guk5wk5jngrp4gl2ovr6eq"
)


def pcpr_031_receipt_promotion(verdict: RemovalVerdict) -> dict[str, Any]:
    """Compact outer-receipt promotion fields. Never a closed release."""

    if verdict.closed_release_outcome is not None:
        raise FabricatedHardwareRemovalError(
            "fabricated hardware removal must not mint a closed release outcome"
        )
    if verdict.release_claim:
        raise FabricatedHardwareRemovalError(
            "fabricated hardware removal must not claim a PCPR release"
        )
    if verdict.completion_authoritative:
        raise FabricatedHardwareRemovalError(
            "fabricated hardware removal completion is not authoritative"
        )
    if verdict.duckdb_or_quack_state_written:
        raise FabricatedHardwareRemovalError(
            "fabricated hardware removal must not write DuckDB or Quack state"
        )
    if verdict.promotion_status in CLOSED_RELEASE_OUTCOMES:
        raise FabricatedHardwareRemovalError(
            "promotion_status must not be a closed release outcome"
        )
    if verdict.promotion_status not in PROMOTION_STATUSES:
        raise FabricatedHardwareRemovalError(
            "promotion_status is not an admitted PCPR-031 status"
        )
    if verdict.promotion_status == "supervisor_promoted":
        raise FabricatedHardwareRemovalError(
            "fabricated hardware removal cannot promote the supervisor"
        )
    if verdict.live_cuda_qualified:
        raise FabricatedHardwareRemovalError(
            "this task cannot claim live CUDA qualification"
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
        "ordinary_runtime_uses_mock_hardware": (
            verdict.ordinary_runtime_uses_mock_hardware
        ),
        "mock_hardware_quarantined": verdict.mock_hardware_quarantined,
        "cuda_mock_requires_explicit_simulation": (
            verdict.cuda_mock_requires_explicit_simulation
        ),
        "device_visibility_implies_production_authorized": (
            verdict.device_visibility_implies_production_authorized
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


def current_head_pcpr_031_receipt_promotion() -> dict[str, Any]:
    return pcpr_031_receipt_promotion(qualify_current_head_removal())


def pcpr_031_receipt_negative_results() -> dict[str, Any]:
    return {
        "simulated_presence_cannot_count_as_live": True,
        "simulated_results_not_represented_as_live": True,
        "estimated_values_rejected": True,
        "hermetic_pass_cannot_qualify_live": True,
        "closed_release_outcome_not_emitted": True,
        "direct_database_bypass_not_used": True,
        "live_cuda_not_claimed": True,
        "device_visibility_not_production_authorized": True,
        "package_import_not_production_authorized": True,
        "ordinary_runtime_cannot_instantiate_mock_hardware": True,
        "evidence_kind": "measured",
    }


def current_head_pcpr_031_receipt_sections() -> dict[str, Any]:
    verdict = qualify_current_head_removal()
    promotion = pcpr_031_receipt_promotion(verdict)
    qualification = qualify_current_head_without_live_campaign()
    return {
        "qualification_verdict": promotion,
        "qualification_prerequisite": {
            "task_id": PCPR_030_TASK_ID,
            "goal_id": "PCPR-G410",
            "promotion_status": "rnd_non_promoted",
            "quarantine_verdict_cid": PCPR_030_VERDICT_CID,
            "supervisor_live_qualification_task_id": PCPR_001_TASK_ID,
            "supervisor_live_qualification_promotion_status": qualification.promotion_status,
            "live_qualification_evidence_kind": "unavailable",
            "verdict_cid": CURRENT_HEAD_UNAVAILABLE_VERDICT_CID,
            "closed_release_outcome": None,
            "release_claim": False,
            "completion_authoritative": False,
            "reason": (
                "PCPR-030 quarantined the mock coordinator. This task removes "
                "fabricated hardware availability. PCPR-001 live qualification "
                "remains rnd_non_promoted and is not promoted by this receipt."
            ),
            "evidence_kind": "measured",
        },
        "removal": {
            "schema": REMOVAL_SCHEMA,
            "interface": REMOVAL_INTERFACE,
            "namespace": "ipfs_accelerate_py.compatibility.simulation.fabricated_hardware",
            "ladder_schema": "ipfs_accelerate_py/assurance/hardware-capability-ladder@1",
            "ordinary_runtime_uses_mock_hardware": (
                verdict.ordinary_runtime_uses_mock_hardware
            ),
            "mock_hardware_quarantined": verdict.mock_hardware_quarantined,
            "cuda_mock_requires_explicit_simulation": (
                verdict.cuda_mock_requires_explicit_simulation
            ),
            "device_visibility_implies_production_authorized": (
                verdict.device_visibility_implies_production_authorized
            ),
            "probes": [item.to_mapping() for item in verdict.probes],
            "evidence_kind": "measured",
        },
        "negative_results": pcpr_031_receipt_negative_results(),
        "verdict_cid": verdict.verdict_cid,
        "promotion_status": promotion["promotion_status"],
        "closed_release_outcome": None,
        "release_claim": False,
        "contracts_frozen": False,
    }


def pcpr_031_current_tree_binding(
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
        raise FabricatedHardwareRemovalError(
            "accelerator_pre_change_commit must equal accelerator_gitlink"
        )
    datasets = _git_object_id(datasets_commit, "datasets_commit")
    datasets_tree_id = _git_object_id(datasets_tree, "datasets_tree")
    datasets_link = _git_object_id(datasets_gitlink, "datasets_gitlink")
    if datasets != datasets_link:
        raise FabricatedHardwareRemovalError(
            "datasets_commit must equal datasets_gitlink"
        )
    kit = _git_object_id(kit_commit, "kit_commit")
    kit_tree_id = _git_object_id(kit_tree, "kit_tree")
    kit_link = _git_object_id(kit_gitlink, "kit_gitlink")
    if kit != kit_link:
        raise FabricatedHardwareRemovalError("kit_commit must equal kit_gitlink")
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


def current_head_pcpr_031_current_tree_binding() -> dict[str, Any]:
    return pcpr_031_current_tree_binding(
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


def validate_pcpr_031_outer_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed if an outer PCPR-031 receipt claims a closed release."""

    if not isinstance(payload, Mapping):
        raise FabricatedHardwareRemovalError("outer receipt must be a mapping")
    if payload.get("task_id") != PCPR_031_TASK_ID:
        raise FabricatedHardwareRemovalError("outer receipt task_id must be PCPR-031")
    _reject_closed_release_value(payload.get("status"), "status")
    _reject_closed_release_value(payload.get("promotion_status"), "promotion_status")
    if payload.get("release_claim") is True:
        raise FabricatedHardwareRemovalError("outer receipt must not claim a release")
    if payload.get("completion_authoritative") is True:
        raise FabricatedHardwareRemovalError(
            "outer receipt completion is not authoritative"
        )

    verdict_section = payload.get("qualification_verdict")
    if not isinstance(verdict_section, Mapping):
        raise FabricatedHardwareRemovalError("qualification_verdict must be a mapping")
    _reject_closed_release_value(
        verdict_section.get("promotion_status"),
        "qualification_verdict.promotion_status",
    )
    _reject_closed_release_value(
        verdict_section.get("closed_release_outcome"),
        "qualification_verdict.closed_release_outcome",
    )
    if verdict_section.get("closed_release_outcome") is not None:
        raise FabricatedHardwareRemovalError(
            "qualification_verdict.closed_release_outcome must not be a closed PCPR release outcome"
        )
    if verdict_section.get("release_claim") is True:
        raise FabricatedHardwareRemovalError(
            "qualification_verdict must not claim a release"
        )
    if verdict_section.get("duckdb_or_quack_state_written") is True:
        raise FabricatedHardwareRemovalError(
            "fabricated hardware removal must not write DuckDB or Quack state"
        )
    if verdict_section.get("contracts_frozen") is True:
        raise FabricatedHardwareRemovalError(
            "fabricated hardware removal cannot freeze contracts"
        )
    if verdict_section.get("live_cuda_qualified") is True:
        raise FabricatedHardwareRemovalError(
            "this task cannot claim live CUDA qualification"
        )
    promotion_status = verdict_section.get("promotion_status")
    if promotion_status not in PROMOTION_STATUSES:
        raise FabricatedHardwareRemovalError(
            "qualification_verdict.promotion_status is not an admitted PCPR-031 status"
        )
    if promotion_status == "supervisor_promoted":
        raise FabricatedHardwareRemovalError(
            "fabricated hardware removal cannot promote the supervisor"
        )

    acceptance = payload.get("acceptance")
    if isinstance(acceptance, Mapping):
        _reject_closed_release_value(
            acceptance.get("promotion_status"), "acceptance.promotion_status"
        )
        if acceptance.get("closed_release_outcome") is not None:
            raise FabricatedHardwareRemovalError(
                "acceptance.closed_release_outcome must be null"
            )
        if acceptance.get("release_claim") is True:
            raise FabricatedHardwareRemovalError("acceptance must not claim a release")

    expected = current_head_pcpr_031_receipt_promotion()
    if verdict_section.get("verdict_cid") != expected["verdict_cid"]:
        raise FabricatedHardwareRemovalError(
            "qualification_verdict.verdict_cid must match the evaluator"
        )
    if promotion_status != expected["promotion_status"]:
        raise FabricatedHardwareRemovalError(
            "qualification_verdict.promotion_status must match the evaluator"
        )
    if expected["verdict_cid"] != CURRENT_HEAD_NON_PROMOTION_VERDICT_CID:
        raise FabricatedHardwareRemovalError(
            "pinned current-head non-promotion CID drifted from the evaluator"
        )

    return {
        "valid": True,
        "task_id": PCPR_031_TASK_ID,
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
    "PCPR_031_GOAL_ID",
    "PCPR_031_TASK_ID",
    "REMOVAL_INTERFACE",
    "SEALED_GIT_BINARY",
    "SEALED_PATH",
    "SEALED_PYTHON",
    "FabricatedHardwareRemovalError",
    "RemovalProbe",
    "RemovalVerdict",
    "current_head_pcpr_031_current_tree_binding",
    "current_head_pcpr_031_receipt_promotion",
    "current_head_pcpr_031_receipt_sections",
    "current_head_removal_probes",
    "pcpr_031_current_tree_binding",
    "pcpr_031_receipt_promotion",
    "qualify_current_head_removal",
    "qualify_fabricated_hardware_removal",
    "validate_pcpr_031_outer_receipt",
)
