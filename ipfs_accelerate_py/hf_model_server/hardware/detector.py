"""
Hardware Detection and Selection

Detect available hardware and select optimal hardware for models.

PCPR-031: detection is a ladder rung, not qualification. Package import,
adapter presence, and device visibility never imply production_authorized.
Missing environments stay typed unavailable (available is None).
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


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
        """Detect CUDA/NVIDIA GPU"""
        try:
            import torch

            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                device = torch.cuda.current_device()
                memory_total = torch.cuda.get_device_properties(device).total_memory / (1024**2)
                memory_allocated = torch.cuda.memory_allocated(device) / (1024**2)
                memory_available = memory_total - memory_allocated
                compute_capability = f"{torch.cuda.get_device_capability(device)[0]}.{torch.cuda.get_device_capability(device)[1]}"

                return HardwareCapability(
                    name="cuda",
                    available=True,
                    device_count=device_count,
                    memory_total_mb=memory_total,
                    memory_available_mb=memory_available,
                    compute_capability=compute_capability,
                    metadata={
                        "torch_version": torch.__version__,
                        "device_visibility_is_not_qualification": True,
                    },
                    declared=True,
                    installed=True,
                    detected=True,
                    canary_passed=None,
                    qualified=False,
                    production_authorized=False,
                    origin="hermetic_observed",
                    live=False,
                    evidence_kind="measured",
                    outcome="Unavailable",
                )
        except ImportError:
            logger.debug("CUDA typed unavailable: torch is not installed")
            return HardwareCapability(
                name="cuda",
                available=None,
                origin="absent",
                evidence_kind="unavailable",
                outcome="Unavailable",
            )
        except Exception as e:
            logger.debug(f"CUDA not available: {e}")

        return HardwareCapability(
            name="cuda",
            available=False,
            declared=True,
            installed=True,
            detected=False,
            production_authorized=False,
            origin="hermetic_observed",
            live=False,
            evidence_kind="measured",
            outcome="Unavailable",
        )

    def _detect_rocm(self) -> HardwareCapability:
        """Detect ROCm/AMD GPU"""
        try:
            import torch

            if hasattr(torch, "hip") and torch.hip.is_available():
                device_count = torch.hip.device_count()
                return HardwareCapability(
                    name="rocm",
                    available=True,
                    device_count=device_count,
                    metadata={
                        "torch_version": torch.__version__,
                        "device_visibility_is_not_qualification": True,
                    },
                    declared=True,
                    installed=True,
                    detected=True,
                    qualified=False,
                    production_authorized=False,
                    origin="hermetic_observed",
                    live=False,
                    evidence_kind="measured",
                    outcome="Unavailable",
                )
        except ImportError:
            return HardwareCapability(
                name="rocm",
                available=None,
                origin="absent",
                evidence_kind="unavailable",
                outcome="Unavailable",
            )
        except Exception as e:
            logger.debug(f"ROCm not available: {e}")

        return HardwareCapability(
            name="rocm",
            available=None,
            origin="absent",
            evidence_kind="unavailable",
            outcome="Unavailable",
        )

    def _detect_mps(self) -> HardwareCapability:
        """Detect Apple Metal Performance Shaders"""
        try:
            import torch

            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return HardwareCapability(
                    name="mps",
                    available=True,
                    device_count=1,
                    metadata={
                        "torch_version": torch.__version__,
                        "device_visibility_is_not_qualification": True,
                    },
                    declared=True,
                    installed=True,
                    detected=True,
                    qualified=False,
                    production_authorized=False,
                    origin="hermetic_observed",
                    live=False,
                    evidence_kind="measured",
                    outcome="Unavailable",
                )
        except ImportError:
            return HardwareCapability(
                name="mps",
                available=None,
                origin="absent",
                evidence_kind="unavailable",
                outcome="Unavailable",
            )
        except Exception as e:
            logger.debug(f"MPS not available: {e}")

        return HardwareCapability(
            name="mps",
            available=None,
            origin="absent",
            evidence_kind="unavailable",
            outcome="Unavailable",
        )

    def _detect_openvino(self) -> HardwareCapability:
        """Detect OpenVINO"""
        try:
            import openvino

            return HardwareCapability(
                name="openvino",
                available=False,
                device_count=0,
                metadata={
                    "version": openvino.__version__,
                    "package_import_is_not_qualification": True,
                },
                declared=True,
                installed=True,
                detected=None,
                qualified=False,
                production_authorized=False,
                origin="declared",
                live=False,
                evidence_kind="measured",
                outcome="Unavailable",
            )
        except ImportError:
            return HardwareCapability(
                name="openvino",
                available=None,
                origin="absent",
                evidence_kind="unavailable",
                outcome="Unavailable",
            )
        except Exception as e:
            logger.debug(f"OpenVINO not available: {e}")

        return HardwareCapability(
            name="openvino",
            available=None,
            origin="absent",
            evidence_kind="unavailable",
            outcome="Unavailable",
        )

    def _detect_qnn(self) -> HardwareCapability:
        """Detect Qualcomm Neural Network SDK"""
        try:
            # QNN detection would go here
            # For now, assume not available
            pass
        except Exception as e:
            logger.debug(f"QNN not available: {e}")

        return HardwareCapability(
            name="qnn",
            available=None,
            origin="absent",
            evidence_kind="unavailable",
            outcome="Unavailable",
            production_authorized=False,
        )

    def _detect_cpu(self) -> HardwareCapability:
        """Detect CPU (always available)"""
        try:
            import psutil

            memory = psutil.virtual_memory()
            return HardwareCapability(
                name="cpu",
                available=True,
                device_count=1,
                memory_total_mb=memory.total / (1024**2),
                memory_available_mb=memory.available / (1024**2),
                declared=True,
                installed=True,
                detected=True,
                qualified=False,
                production_authorized=False,
                origin="declared",
                live=False,
                evidence_kind="measured",
                outcome="Unavailable",
            )
        except Exception as e:
            # A running process implies a declared CPU host, not qualification.
            del e
            return HardwareCapability(
                name="cpu",
                available=True,
                device_count=1,
                declared=True,
                installed=True,
                detected=None,
                qualified=False,
                production_authorized=False,
                origin="declared",
                live=False,
                evidence_kind="measured",
                outcome="Unavailable",
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

    def get_best_hardware(
        self, supported_hardware: List[str], preferred_order: List[str] = None
    ) -> Optional[str]:
        """
        Select best available hardware

        Args:
            supported_hardware: Hardware supported by the model
            preferred_order: Preferred hardware order

        Returns:
            Best hardware name or None
        """
        if preferred_order is None:
            preferred_order = ["cuda", "rocm", "mps", "openvino", "qnn", "cpu"]

        # Find first preferred hardware that is both available and supported
        for hardware in preferred_order:
            if hardware in supported_hardware and self.is_available(hardware):
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
        self, model_info: Dict, preferred_order: List[str] = None
    ) -> Tuple[Optional[str], str]:
        """
        Select optimal hardware for a model

        Args:
            model_info: Model information including supported_hardware
            preferred_order: Preferred hardware order

        Returns:
            Tuple of (hardware_name, reason)
        """
        supported = model_info.get("supported_hardware", ["cpu"])

        # Get best available hardware
        hardware = self.detector.get_best_hardware(supported, preferred_order)

        if hardware:
            reason = (
                f"Selected {hardware} (detected and supported; "
                "detection is not production_authorized)"
            )
            return hardware, reason
        else:
            reason = (
                "No suitable detected hardware; falling back to declared CPU "
                "baseline (not production_authorized)"
            )
            return "cpu", reason

    def track_load(self, hardware: str, delta: int = 1):
        """Track hardware load (increment/decrement)"""
        current = self.load_tracker.get(hardware, 0)
        self.load_tracker[hardware] = max(0, current + delta)

    def get_load(self, hardware: str) -> int:
        """Get current load for hardware"""
        return self.load_tracker.get(hardware, 0)
