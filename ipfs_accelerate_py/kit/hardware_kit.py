"""
Hardware Kit - Core Hardware Detection and Management

This module provides core hardware detection and management operations.
It can be used by both the unified CLI and MCP server.
"""

import json
import logging
import platform
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


_DEFAULT_HARDWARE_KIT: Optional["HardwareKit"] = None


def _get_default_hardware_kit() -> "HardwareKit":
    global _DEFAULT_HARDWARE_KIT
    if _DEFAULT_HARDWARE_KIT is None:
        _DEFAULT_HARDWARE_KIT = HardwareKit()
    return _DEFAULT_HARDWARE_KIT


@dataclass
class HardwareInfo:
    """Hardware information."""

    cpu: Dict[str, Any] = field(default_factory=dict)
    gpu: Dict[str, Any] = field(default_factory=dict)
    memory: Dict[str, Any] = field(default_factory=dict)
    accelerators: Dict[str, Any] = field(default_factory=dict)
    platform_info: Dict[str, Any] = field(default_factory=dict)


class HardwareKit:
    """
    Core hardware operations module.

    Provides hardware detection and management functionality that can be
    used by CLI, MCP tools, or directly in Python code.
    """

    def __init__(self):
        """Initialize Hardware Kit."""
        pass

    def get_platform_info(self) -> Dict[str, Any]:
        """
        Get platform information.

        Returns:
            Dictionary with platform info
        """
        try:
            return {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "python_version": platform.python_version(),
            }
        except Exception as e:
            logger.warning(f"Failed to get platform info: {e}")
            return {}

    def get_cpu_info(self) -> Dict[str, Any]:
        """
        Get CPU information.

        Returns:
            Dictionary with CPU info
        """
        import os

        cpu_info = {
            "count": os.cpu_count() or 0,
            "architecture": platform.machine(),
        }

        # Try to get more detailed CPU info
        try:
            if platform.system() == "Linux":
                with open("/proc/cpuinfo", "r") as f:
                    cpuinfo = f.read()
                    # Extract model name
                    for line in cpuinfo.split("\n"):
                        if "model name" in line:
                            cpu_info["model"] = line.split(":")[1].strip()
                            break
        except Exception as e:
            logger.debug(f"Could not get detailed CPU info: {e}")

        return cpu_info

    def get_memory_info(self) -> Dict[str, Any]:
        """
        Get memory information.

        Returns:
            Dictionary with memory info
        """
        memory_info = {}

        try:
            if platform.system() == "Linux":
                with open("/proc/meminfo", "r") as f:
                    meminfo = f.read()
                    for line in meminfo.split("\n"):
                        if "MemTotal" in line:
                            # Extract total memory in KB and convert to GB
                            total_kb = int(line.split()[1])
                            memory_info["total_gb"] = round(total_kb / (1024 * 1024), 2)
                        elif "MemAvailable" in line:
                            available_kb = int(line.split()[1])
                            memory_info["available_gb"] = round(available_kb / (1024 * 1024), 2)
        except Exception as e:
            logger.debug(f"Could not get memory info: {e}")

        return memory_info

    def detect_cuda(self) -> Dict[str, Any]:
        """
        Detect CUDA GPUs.

        PCPR-031: nvidia-smi or torch.cuda.is_available is device visibility,
        not production_authorized. Missing tools stay typed unavailable.
        """
        from ipfs_accelerate_py.assurance.hardware_capability_ladder import (
            attach_ladder,
            from_device_visibility,
            from_measured_absence,
            stamp_consolidation,
            unavailable_backend,
        )

        cuda_info = unavailable_backend("cuda", devices=[])
        cuda_info["declared"] = True

        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0 and result.stdout.strip():
                devices = []
                for line in result.stdout.strip().split("\n"):
                    if line:
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) >= 2:
                            devices.append(
                                {
                                    "name": parts[0],
                                    "memory": parts[1] if len(parts) > 1 else "Unknown",
                                    "driver": parts[2] if len(parts) > 2 else "Unknown",
                                }
                            )
                if devices:
                    cuda_info = from_device_visibility(
                        "cuda",
                        devices=devices,
                        extra={"probe": "nvidia-smi"},
                    )
        except FileNotFoundError:
            logger.debug("nvidia-smi not on PATH; CUDA probe remains typed unavailable")
        except subprocess.TimeoutExpired as e:
            logger.debug(f"CUDA nvidia-smi timed out: {e}")

        try:
            import torch

            cuda_info["installed"] = True
            cuda_info["pytorch_version"] = torch.__version__
            if torch.cuda.is_available() is True:
                cuda_info = from_device_visibility(
                    "cuda",
                    devices=cuda_info.get("devices") or [],
                    extra={
                        "probe": "torch.cuda.is_available",
                        "pytorch_version": torch.__version__,
                        "cuda_version": torch.version.cuda,
                    },
                )
                cuda_info["pytorch_version"] = torch.__version__
                cuda_info["cuda_version"] = torch.version.cuda
                if not cuda_info.get("devices"):
                    for i in range(torch.cuda.device_count()):
                        cuda_info["devices"].append(
                            {"name": torch.cuda.get_device_name(i), "index": i}
                        )
            elif torch.cuda.is_available() is False and cuda_info.get("detected") is not True:
                cuda_info = from_measured_absence(
                    "cuda",
                    extra={
                        "probe": "torch.cuda.is_available",
                        "pytorch_version": torch.__version__,
                    },
                )
                cuda_info["devices"] = []
                cuda_info["pytorch_version"] = torch.__version__
        except ImportError:
            if cuda_info.get("detected") is not True:
                cuda_info.setdefault("installed", None)
        except Exception as e:
            logger.debug(f"PyTorch CUDA detection failed: {e}")

        return stamp_consolidation(attach_ladder(cuda_info))

    def detect_rocm(self) -> Dict[str, Any]:
        """
        Detect ROCm GPUs (AMD).

        rocm-smi visibility is detection, not production_authorized.
        """
        from ipfs_accelerate_py.assurance.hardware_capability_ladder import (
            attach_ladder,
            from_device_visibility,
            stamp_consolidation,
            unavailable_backend,
        )

        rocm_info = unavailable_backend("rocm", devices=[])
        rocm_info["declared"] = True

        try:
            result = subprocess.run(
                ["rocm-smi", "--showproductname"], capture_output=True, text=True, timeout=5
            )

            if result.returncode == 0 and result.stdout.strip():
                devices = []
                for line in result.stdout.strip().split("\n"):
                    if "GPU" in line:
                        devices.append({"info": line.strip()})
                if devices:
                    rocm_info = from_device_visibility(
                        "rocm",
                        devices=devices,
                        extra={"probe": "rocm-smi"},
                    )
        except FileNotFoundError:
            logger.debug("rocm-smi not on PATH; ROCm remains typed unavailable")
        except subprocess.TimeoutExpired as e:
            logger.debug(f"ROCm not detected: {e}")

        return stamp_consolidation(attach_ladder(rocm_info))

    def detect_metal(self) -> Dict[str, Any]:
        """
        Detect Metal support (Apple Silicon).

        Returns:
            Dictionary with Metal info
        """
        from ipfs_accelerate_py.assurance.hardware_capability_ladder import (
            attach_ladder,
            from_platform_presence,
            stamp_consolidation,
            unavailable_backend,
        )

        if platform.system() == "Darwin":
            metal_info = from_platform_presence(
                "metal",
                platform_name="macOS",
                extra={
                    "machine": platform.machine(),
                    "apple_silicon": platform.machine() == "arm64",
                    "platform_presence_is_not_qualification": True,
                },
            )
            return stamp_consolidation(attach_ladder(metal_info))

        return stamp_consolidation(attach_ladder(unavailable_backend("metal")))

    def detect_webgpu(self) -> Dict[str, Any]:
        """
        Detect WebGPU support.

        Browser absence stays typed unavailable, never fabricated False.
        """
        from ipfs_accelerate_py.assurance.hardware_capability_ladder import (
            stamp_consolidation,
            unavailable_backend,
        )

        return stamp_consolidation(
            unavailable_backend(
                "webgpu",
                note="WebGPU requires a browser environment",
            )
        )

    def detect_webnn(self) -> Dict[str, Any]:
        """
        Detect WebNN support.

        Browser absence stays typed unavailable, never fabricated False.
        """
        from ipfs_accelerate_py.assurance.hardware_capability_ladder import (
            stamp_consolidation,
            unavailable_backend,
        )

        return stamp_consolidation(
            unavailable_backend(
                "webnn",
                note="WebNN requires a browser environment",
            )
        )

    def get_hardware_info(self, include_detailed: bool = False) -> HardwareInfo:
        """
        Get comprehensive hardware information.

        Args:
            include_detailed: Include detailed hardware information

        Returns:
            HardwareInfo object
        """
        hardware_info = HardwareInfo()

        # Get platform info
        hardware_info.platform_info = self.get_platform_info()

        # Get CPU info
        hardware_info.cpu = self.get_cpu_info()

        # Get memory info
        hardware_info.memory = self.get_memory_info()

        # Detect accelerators. Typed unavailable backends stay in the map;
        # omission must not be read as measured absence or qualification.
        accelerators = {
            "cuda": self.detect_cuda(),
            "rocm": self.detect_rocm(),
            "metal": self.detect_metal(),
        }
        if include_detailed:
            accelerators["webgpu"] = self.detect_webgpu()
            accelerators["webnn"] = self.detect_webnn()

        hardware_info.accelerators = accelerators

        return hardware_info

    def test_hardware(self, accelerator: str = "all", test_level: str = "basic") -> Dict[str, Any]:
        """
        Test hardware accelerators.

        Args:
            accelerator: Accelerator to test (cuda, cpu, webgpu, webnn, all)
            test_level: Level of testing (basic, comprehensive)

        Returns:
            Dictionary with test results
        """
        results = {"tested": accelerator, "level": test_level, "tests": {}}

        if accelerator in ["cuda", "all"]:
            results["tests"]["cuda"] = self._test_cuda(test_level)

        if accelerator in ["cpu", "all"]:
            results["tests"]["cpu"] = self._test_cpu(test_level)

        return results

    def _test_cuda(self, test_level: str) -> Dict[str, Any]:
        """Test CUDA functionality. A passing canary is not production_authorized."""
        from ipfs_accelerate_py.assurance.hardware_capability_ladder import (
            from_canary,
            from_measured_absence,
            stamp_consolidation,
            unavailable_backend,
        )

        result = stamp_consolidation(unavailable_backend("cuda"))
        result["tests_passed"] = False
        result["canary_passed"] = None

        try:
            import torch

            result["installed"] = True
            if torch.cuda.is_available() is True:
                canary_passed = False
                if test_level == "basic":
                    x = torch.ones(10, device="cuda")
                    del x
                    canary_passed = True
                elif test_level == "comprehensive":
                    x = torch.randn(1000, 1000, device="cuda")
                    y = torch.matmul(x, x)
                    del x, y
                    canary_passed = True
                result = from_canary(
                    "cuda",
                    passed=canary_passed,
                    extra={
                        "probe": "torch.cuda.canary",
                        "test_level": test_level,
                        "pytorch_version": torch.__version__,
                    },
                )
                result["tests_passed"] = canary_passed
                result["device_count"] = torch.cuda.device_count()
            elif torch.cuda.is_available() is False:
                result = from_measured_absence(
                    "cuda",
                    extra={"probe": "torch.cuda.is_available"},
                )
                result["tests_passed"] = False
        except ImportError:
            result["error"] = "torch_unavailable"
        except Exception as e:
            result["error"] = str(e)

        return stamp_consolidation(result)

    def _test_cpu(self, test_level: str) -> Dict[str, Any]:
        """Live CPU canary. Passing execution is not production_authorized.

        PCPR-037: stdlib CPU kernel, cancellation, timeout, cleanup, and
        fail-closed resource admission. Numpy is not required. CUDA and
        model/provider paths stay typed unavailable.
        """
        from ipfs_accelerate_py.assurance.cpu_execution import qualify_live_cpu_execution

        report = qualify_live_cpu_execution(test_level=test_level)
        report["tests_passed"] = bool(report.get("cpu_execution_qualified") is True)
        report["production_authorized"] = False
        report["qualified"] = False
        return report

    def recommend_hardware(
        self, model_name: str, task: str = "inference", consider_available_only: bool = True
    ) -> Dict[str, Any]:
        """
        Get hardware recommendations for a model.

        Args:
            model_name: Model name
            task: Task type (inference, training, fine-tuning)
            consider_available_only: Only consider available hardware

        Returns:
            Dictionary with recommendations
        """
        recommendations = {"model": model_name, "task": task, "recommendations": []}

        hardware = self.get_hardware_info()
        cuda_report = hardware.accelerators.get("cuda") or {}
        cuda_detected = cuda_report.get("available") is True
        recommendations["production_authorized"] = False
        recommendations["live"] = False
        recommendations["outcome"] = "Unavailable"
        recommendations["advisory"] = True

        if "large" in model_name.lower() or "xl" in model_name.lower():
            if cuda_detected:
                recommendations["recommendations"].append(
                    {
                        "accelerator": "cuda",
                        "reason": "Large model benefits from GPU acceleration (advisory; not production_authorized)",
                        "priority": 1,
                        "available": True,
                        "production_authorized": False,
                    }
                )
            elif not consider_available_only:
                recommendations["recommendations"].append(
                    {
                        "accelerator": "cuda",
                        "reason": "Large model requires GPU for reasonable performance (advisory; CUDA evidence typed unavailable)",
                        "priority": 1,
                        "available": cuda_report.get("available"),
                        "production_authorized": False,
                    }
                )

        recommendations["recommendations"].append(
            {
                "accelerator": "cpu",
                "reason": "Declared CPU host fallback (not production_authorized)",
                "priority": 10,
                "available": True,
                "production_authorized": False,
            }
        )

        return recommendations


# Convenience functions


def get_hardware_kit() -> HardwareKit:
    """
    Get a HardwareKit instance.

    Returns:
        HardwareKit instance
    """
    return HardwareKit()


def get_info() -> Dict[str, Any]:
    """Module-level wrapper for unified tool registry."""

    info = _get_default_hardware_kit().get_hardware_info(include_detailed=False)
    return {
        "cpu": info.cpu,
        "gpu": info.gpu,
        "memory": info.memory,
        "accelerators": info.accelerators,
        "platform_info": info.platform_info,
    }


def test() -> Dict[str, Any]:
    """Module-level wrapper for unified tool registry."""

    return _get_default_hardware_kit().test_hardware(accelerator="all", test_level="basic")


def recommend(task_type: str) -> Dict[str, Any]:
    """Module-level wrapper for unified tool registry.

    The registry schema currently provides `task_type` (inference/training/etc).
    We translate that into a basic accelerator recommendation using detected
    hardware.
    """

    task = (task_type or "").strip().lower() or "inference"
    hw = _get_default_hardware_kit().get_hardware_info(include_detailed=False)
    cuda_report = hw.accelerators.get("cuda") or {}
    has_cuda = cuda_report.get("available") is True

    recs: List[Dict[str, Any]] = []
    if has_cuda and task in {"training", "fine-tuning", "finetuning", "inference"}:
        recs.append(
            {
                "accelerator": "cuda",
                "reason": "CUDA GPU detected (advisory; not production_authorized)",
                "priority": 1,
                "available": True,
                "production_authorized": False,
            }
        )

    recs.append(
        {
            "accelerator": "cpu",
            "reason": "Declared CPU host fallback (not production_authorized)",
            "priority": 10,
            "available": True,
            "production_authorized": False,
        }
    )

    return {
        "task_type": task_type,
        "recommendations": recs,
        "hardware": {"accelerators": hw.accelerators},
        "production_authorized": False,
        "live": False,
        "outcome": "Unavailable",
        "advisory": True,
    }


__all__ = [
    "HardwareKit",
    "HardwareInfo",
    "get_hardware_kit",
    "get_info",
    "test",
    "recommend",
]
