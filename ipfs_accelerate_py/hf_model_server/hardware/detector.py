"""
Hardware Detection and Selection

Detect available hardware and select optimal hardware for models.

PCPR-031/PCPR-034: detection is a ladder rung, not qualification. Package
import, adapter presence, and device visibility never imply
production_authorized. Production execution is admitted only when the
canonical ladder grants that rung. This detector never grants it.
Missing environments stay typed unavailable (available is None).
"""

import logging
from typing import Any, Dict, List, Mapping, Optional, Tuple
from dataclasses import dataclass

from ipfs_accelerate_py.assurance.hardware_capability_ladder import (
    admit_production_execution,
    attach_ladder,
    declared_cpu_baseline,
    from_device_visibility,
    from_measured_absence,
    from_package_import,
    refused_production_execution,
    stamp_consolidation,
    unavailable_backend,
)


logger = logging.getLogger(__name__)


@dataclass
class HardwareCapability:
    """Hardware capability information.

    ``available`` is a detection flag only. ``production_authorized`` is never
    implied by visibility, import, or configuration.
    """

    name: str
    available: Optional[bool] = None
    device_count: int = 0
    memory_total_mb: float = 0
    memory_available_mb: float = 0
    compute_capability: Optional[str] = None
    metadata: Dict = None
    declared: bool = False
    installed: Optional[bool] = None
    detected: Optional[bool] = None
    canary_passed: Optional[bool] = None
    model_compatible: Optional[bool] = None
    resource_sufficient: Optional[bool] = None
    qualified: bool = False
    production_authorized: bool = False
    origin: str = "absent"
    live: bool = False
    evidence_kind: str = "unavailable"
    outcome: str = "Unavailable"

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        self.production_authorized = False
        self.qualified = False
        self.live = False

    @classmethod
    def from_ladder(
        cls, report: Mapping[str, Any], **overrides: Any
    ) -> "HardwareCapability":
        """Construct a capability from a canonical ladder report."""

        payload = stamp_consolidation(dict(report))
        payload.update(overrides)
        attach_ladder(payload)
        metadata = dict(payload.get("extra") or payload.get("metadata") or {})
        for key in (
            "package_import_is_not_qualification",
            "device_visibility_is_not_qualification",
            "adapter_presence_is_not_qualification",
            "configuration_is_not_qualification",
            "testing_defaults_are_not_qualification",
            "consolidation_schema",
            "consolidation_interface",
            "consolidation_task_id",
            "ladder_assessment",
            "torch_version",
            "version",
            "package",
            "probe",
            "devices",
        ):
            if key in payload and key not in metadata:
                metadata[key] = payload[key]
        return cls(
            name=str(payload.get("backend") or payload.get("name") or "unknown"),
            available=payload.get("available"),
            device_count=int(payload.get("device_count") or 0),
            memory_total_mb=float(payload.get("memory_total_mb") or 0),
            memory_available_mb=float(payload.get("memory_available_mb") or 0),
            compute_capability=payload.get("compute_capability"),
            metadata=metadata,
            declared=bool(payload.get("declared")),
            installed=payload.get("installed"),
            detected=payload.get("detected"),
            canary_passed=payload.get("canary_passed"),
            model_compatible=payload.get("model_compatible"),
            resource_sufficient=payload.get("resource_sufficient"),
            qualified=False,
            production_authorized=False,
            origin=str(payload.get("origin") or "absent"),
            live=False,
            evidence_kind=str(payload.get("evidence_kind") or "unavailable"),
            outcome=str(payload.get("outcome") or "Unavailable"),
        )

    def to_ladder(self) -> Dict[str, Any]:
        """Export this capability as a canonical ladder report."""

        report = {
            "backend": self.name,
            "available": self.available,
            "declared": self.declared,
            "installed": self.installed,
            "detected": self.detected,
            "canary_passed": self.canary_passed,
            "model_compatible": self.model_compatible,
            "resource_sufficient": self.resource_sufficient,
            "qualified": False,
            "production_authorized": False,
            "origin": self.origin,
            "live": False,
            "evidence_kind": self.evidence_kind,
            "outcome": self.outcome,
            "device_count": self.device_count,
            "memory_total_mb": self.memory_total_mb,
            "memory_available_mb": self.memory_available_mb,
            "compute_capability": self.compute_capability,
            "metadata": dict(self.metadata or {}),
        }
        return stamp_consolidation(report)


class HardwareDetector:
    """
    Detect available hardware capabilities

    Supports:
    - CUDA (NVIDIA GPUs)
    - ROCm (AMD GPUs)
    - MPS (Apple Silicon)
    - OpenVINO (Intel)
    - QNN (Qualcomm)
    - CPU (always available)
    """

    def __init__(self):
        self.capabilities: Dict[str, HardwareCapability] = {}
        self._detect_all()

    def _detect_all(self):
        """Detect all hardware"""
        self.capabilities = {
            "cuda": self._detect_cuda(),
            "rocm": self._detect_rocm(),
            "mps": self._detect_mps(),
            "openvino": self._detect_openvino(),
            "qnn": self._detect_qnn(),
            "cpu": self._detect_cpu(),
        }

        detected = [
            name for name, cap in self.capabilities.items() if cap.available is True
        ]
        logger.info(
            "Detected hardware (not production_authorized): "
            f"{', '.join(detected) if detected else 'none'}"
        )

    def _detect_cuda(self) -> HardwareCapability:
        """Detect CUDA/NVIDIA GPU. Detection is not production_authorized."""
        try:
            import torch

            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                device = torch.cuda.current_device()
                memory_total = torch.cuda.get_device_properties(device).total_memory / (1024**2)
                memory_allocated = torch.cuda.memory_allocated(device) / (1024**2)
                memory_available = memory_total - memory_allocated
                compute_capability = f"{torch.cuda.get_device_capability(device)[0]}.{torch.cuda.get_device_capability(device)[1]}"
                report = from_device_visibility(
                    "cuda",
                    devices=[{"index": device, "name": torch.cuda.get_device_name(device)}],
                    extra={
                        "probe": "torch.cuda.is_available",
                        "torch_version": torch.__version__,
                    },
                )
                return HardwareCapability.from_ladder(
                    report,
                    device_count=device_count,
                    memory_total_mb=memory_total,
                    memory_available_mb=memory_available,
                    compute_capability=compute_capability,
                )
        except ImportError:
            logger.debug("CUDA typed unavailable: torch is not installed")
            return HardwareCapability.from_ladder(unavailable_backend("cuda"))
        except Exception as e:
            logger.debug(f"CUDA not available: {e}")

        return HardwareCapability.from_ladder(from_measured_absence("cuda"))

    def _detect_rocm(self) -> HardwareCapability:
        """Detect ROCm/AMD GPU. Detection is not production_authorized."""
        try:
            import torch

            if hasattr(torch, "hip") and torch.hip.is_available():
                device_count = torch.hip.device_count()
                report = from_device_visibility(
                    "rocm",
                    devices=[{"count": device_count}],
                    extra={
                        "probe": "torch.hip.is_available",
                        "torch_version": torch.__version__,
                    },
                )
                return HardwareCapability.from_ladder(report, device_count=device_count)
        except ImportError:
            return HardwareCapability.from_ladder(unavailable_backend("rocm"))
        except Exception as e:
            logger.debug(f"ROCm not available: {e}")

        return HardwareCapability.from_ladder(unavailable_backend("rocm"))

    def _detect_mps(self) -> HardwareCapability:
        """Detect Apple Metal Performance Shaders. Detection is not production_authorized."""
        try:
            import torch

            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                report = from_device_visibility(
                    "mps",
                    devices=[{"count": 1}],
                    extra={
                        "probe": "torch.backends.mps.is_available",
                        "torch_version": torch.__version__,
                    },
                )
                return HardwareCapability.from_ladder(report, device_count=1)
        except ImportError:
            return HardwareCapability.from_ladder(unavailable_backend("mps"))
        except Exception as e:
            logger.debug(f"MPS not available: {e}")

        return HardwareCapability.from_ladder(unavailable_backend("mps"))

    def _detect_openvino(self) -> HardwareCapability:
        """Detect OpenVINO. Package import is not production_authorized."""
        try:
            import openvino

            report = from_package_import(
                "openvino",
                package="openvino",
                version=getattr(openvino, "__version__", None),
            )
            return HardwareCapability.from_ladder(report)
        except ImportError:
            return HardwareCapability.from_ladder(unavailable_backend("openvino"))
        except Exception as e:
            logger.debug(f"OpenVINO not available: {e}")

        return HardwareCapability.from_ladder(unavailable_backend("openvino"))

    def _detect_qnn(self) -> HardwareCapability:
        """Detect Qualcomm Neural Network SDK. Absence stays typed unavailable."""
        try:
            # QNN detection would go here. Missing evidence stays typed unavailable.
            pass
        except Exception as e:
            logger.debug(f"QNN not available: {e}")

        return HardwareCapability.from_ladder(unavailable_backend("qnn"))

    def _detect_cpu(self) -> HardwareCapability:
        """Detect CPU as a declared process host. Not production_authorized."""
        try:
            import psutil

            memory = psutil.virtual_memory()
            report = declared_cpu_baseline(
                cores=None,
                memory_total_mb=memory.total / (1024**2),
                memory_available_mb=memory.available / (1024**2),
            )
            return HardwareCapability.from_ladder(
                report,
                device_count=1,
                memory_total_mb=memory.total / (1024**2),
                memory_available_mb=memory.available / (1024**2),
                detected=True,
            )
        except Exception as e:
            # A running process implies a declared CPU host, not qualification.
            del e
            return HardwareCapability.from_ladder(
                declared_cpu_baseline(cores=None),
                device_count=1,
            )

    def get_capability(self, hardware: str) -> Optional[HardwareCapability]:
        """Get capability for specific hardware"""
        return self.capabilities.get(hardware)

    def is_available(self, hardware: str) -> bool:
        """Check if hardware is detected. Detection is not production_authorized."""
        cap = self.capabilities.get(hardware)
        return bool(cap is not None and cap.available is True)

    def is_production_authorized(self, hardware: str) -> bool:
        """Production execution requires the full ladder. This detector never grants it."""
        cap = self.capabilities.get(hardware)
        return bool(cap is not None and cap.production_authorized is True)

    def get_available_hardware(self) -> List[str]:
        """Get list of detected hardware. Detection is not qualification."""
        return [name for name, cap in self.capabilities.items() if cap.available is True]

    def get_ladder_map(self) -> Dict[str, Dict[str, Any]]:
        """Canonical ladder reports for every known backend."""
        return {name: cap.to_ladder() for name, cap in self.capabilities.items()}

    def get_best_hardware(
        self, supported_hardware: List[str], preferred_order: List[str] = None
    ) -> Optional[str]:
        """
        Select best detected hardware. Detection is advisory only.

        Production execution must use get_best_production_hardware.
        """
        if preferred_order is None:
            preferred_order = ["cuda", "rocm", "mps", "openvino", "qnn", "cpu"]

        for hardware in preferred_order:
            if hardware in supported_hardware and self.is_available(hardware):
                return hardware

        return None

    def get_best_production_hardware(
        self, supported_hardware: List[str], preferred_order: List[str] = None
    ) -> Optional[str]:
        """Select hardware that is production_authorized. This detector never grants it."""
        if preferred_order is None:
            preferred_order = ["cuda", "rocm", "mps", "openvino", "qnn", "cpu"]

        for hardware in preferred_order:
            if hardware in supported_hardware and self.is_production_authorized(hardware):
                return hardware
        return None


class HardwareSelector:
    """
    Intelligent hardware selection for models

    Considers:
    - Hardware availability
    - Model support
    - Current load
    - Memory constraints
    """

    def __init__(self, detector: HardwareDetector):
        self.detector = detector
        self.load_tracker: Dict[str, int] = {}  # Track active models per hardware

    def select_hardware(
        self,
        model_info: Dict,
        preferred_order: List[str] = None,
        *,
        purpose: str = "production",
    ) -> Tuple[Optional[str], str]:
        """
        Select hardware for a model.

        Production purpose requires production_authorized and never falls back
        to a detected or declared CPU host. Advisory purpose may return a
        detected backend and never claims production_authorized.
        """
        supported = model_info.get("supported_hardware", ["cpu"])
        if purpose != "production":
            hardware = self.detector.get_best_hardware(supported, preferred_order)
            if hardware:
                reason = (
                    f"Advisory selection of {hardware} (detected and supported; "
                    "detection is not production_authorized)"
                )
                return hardware, reason
            return None, (
                "No suitable detected hardware for advisory selection; "
                "declared CPU is not production_authorized"
            )

        hardware = self.detector.get_best_production_hardware(
            supported, preferred_order
        )
        if hardware:
            reason = f"Selected {hardware} because production_authorized"
            return hardware, reason
        refusal = refused_production_execution(
            "unspecified",
            reason="no_production_authorized_hardware",
        )
        return None, (
            "No production_authorized hardware; detection, package import, "
            f"and declared CPU cannot execute production work ({refusal['code']})"
        )

    def admit_production_execution(self, hardware: Optional[str]) -> Dict[str, Any]:
        """Fail closed unless the named backend is production_authorized."""
        if not hardware:
            return refused_production_execution(
                "unspecified", reason="production_hardware_absent"
            )
        cap = self.detector.get_capability(hardware)
        if cap is None:
            return refused_production_execution(
                hardware, reason="backend_not_in_detector"
            )
        return admit_production_execution(cap.to_ladder())

    def track_load(self, hardware: str, delta: int = 1):
        """Track hardware load (increment/decrement)"""
        current = self.load_tracker.get(hardware, 0)
        self.load_tracker[hardware] = max(0, current + delta)

    def get_load(self, hardware: str) -> int:
        """Get current load for hardware"""
        return self.load_tracker.get(hardware, 0)
