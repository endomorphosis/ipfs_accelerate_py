#!/usr/bin/env python3
"""
Hardware detection module for IPFS Accelerate.

This module provides utilities for detecting hardware platforms
and skipping tests on unsupported platforms.
"""

import logging
import platform
import warnings
from typing import Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _collect_cuda_devices(torch_module: Any, device_count: int) -> list[dict[str, Any]]:
    """Collect CUDA device metadata for already-probed CUDA devices."""
    return [
        {
            "name": torch_module.cuda.get_device_name(i),
            "capability": torch_module.cuda.get_device_capability(i),
        }
        for i in range(device_count)
    ]


def _detect_cuda_platform(torch_module: Any) -> dict[str, Any]:
    """Detect CUDA availability while suppressing known unsupported-capability noise."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*cuda capability.*",
            category=UserWarning,
        )
        cuda_available = torch_module.cuda.is_available()
        cuda_info: dict[str, Any] = {"available": cuda_available}
        if cuda_available:
            device_count = torch_module.cuda.device_count()
            cuda_info["device_count"] = device_count
            cuda_info["devices"] = _collect_cuda_devices(torch_module, device_count)
        return cuda_info


def detect_hardware() -> Dict[str, Any]:
    """
    Detect available hardware platforms.

    Returns:
        Dictionary with hardware platform information.
    """
    result = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "platforms": {
            "cpu": {"available": True},
            "cuda": {"available": False},
            "rocm": {"available": False},
            "mps": {"available": False},
            "webgpu": {"available": False},
            "webnn": {"available": False},
            "openvino": {"available": False},
            "qnn": {"available": False},
        },
    }

    try:
        import torch

        result["platforms"]["cuda"].update(_detect_cuda_platform(torch))
    except ImportError:
        logger.debug("PyTorch not available, skipping CUDA detection")

    try:
        import torch

        if hasattr(torch, "version") and hasattr(torch.version, "hip") and torch.version.hip is not None:
            result["platforms"]["rocm"]["available"] = True
            result["platforms"]["rocm"]["version"] = torch.version.hip
    except ImportError:
        logger.debug("PyTorch not available, skipping ROCm detection")

    try:
        import torch

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            result["platforms"]["mps"]["available"] = True
    except ImportError:
        logger.debug("PyTorch not available, skipping MPS detection")

    try:
        import openvino

        result["platforms"]["openvino"]["available"] = True
        result["platforms"]["openvino"]["version"] = openvino.__version__
    except ImportError:
        logger.debug("OpenVINO not available")

    try:
        import qti

        result["platforms"]["qnn"]["available"] = True
    except ImportError:
        try:
            import qnn_wrapper

            result["platforms"]["qnn"]["available"] = True
        except ImportError:
            logger.debug("QNN not available")

    result["platforms"]["webgpu"]["available"] = False
    result["platforms"]["webnn"]["available"] = False

    return result


def setup_platform(platform_name: str) -> Any:
    """
    Set up a specific hardware platform.

    Args:
        platform_name: Name of the platform to set up

    Returns:
        Platform-specific object or None if setup fails
    """
    if platform_name == "cuda":
        try:
            import torch

            if torch.cuda.is_available():
                return torch.device("cuda")
        except ImportError:
            logger.warning("PyTorch not available, cannot set up CUDA")

    elif platform_name == "rocm":
        try:
            import torch

            if hasattr(torch, "version") and hasattr(torch.version, "hip") and torch.version.hip is not None:
                return torch.device("cuda")
        except ImportError:
            logger.warning("PyTorch not available, cannot set up ROCm")

    elif platform_name == "mps":
        try:
            import torch

            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
        except ImportError:
            logger.warning("PyTorch not available, cannot set up MPS")

    elif platform_name == "cpu":
        try:
            import torch

            return torch.device("cpu")
        except ImportError:
            logger.warning("PyTorch not available, returning 'cpu' string")
            return "cpu"

    elif platform_name == "openvino":
        try:
            import openvino as ov

            return ov.Core()
        except ImportError:
            logger.warning("OpenVINO not available, cannot set up OpenVINO")

    elif platform_name == "qnn":
        try:
            import qti

            return qti
        except ImportError:
            try:
                import qnn_wrapper

                return qnn_wrapper
            except ImportError:
                logger.warning("QNN not available, cannot set up QNN")

    elif platform_name in ["webgpu", "webnn"]:
        logger.warning("%s requires browser integration", platform_name)

    else:
        logger.warning("Unknown platform: %s", platform_name)

    return None


# Pytest skip decorators

def skip_if_no_cuda(func):
    import pytest

    hardware_info = detect_hardware()
    return pytest.mark.skipif(
        not hardware_info["platforms"]["cuda"]["available"],
        reason="CUDA not available",
    )(func)


def skip_if_no_rocm(func):
    import pytest

    hardware_info = detect_hardware()
    return pytest.mark.skipif(
        not hardware_info["platforms"]["rocm"]["available"],
        reason="ROCm not available",
    )(func)


def skip_if_no_mps(func):
    import pytest

    hardware_info = detect_hardware()
    return pytest.mark.skipif(
        not hardware_info["platforms"]["mps"]["available"],
        reason="MPS not available",
    )(func)


def skip_if_no_openvino(func):
    import pytest

    hardware_info = detect_hardware()
    return pytest.mark.skipif(
        not hardware_info["platforms"]["openvino"]["available"],
        reason="OpenVINO not available",
    )(func)


def skip_if_no_qnn(func):
    import pytest

    hardware_info = detect_hardware()
    return pytest.mark.skipif(
        not hardware_info["platforms"]["qnn"]["available"],
        reason="QNN not available",
    )(func)


def skip_if_no_webgpu(func):
    import pytest

    hardware_info = detect_hardware()
    return pytest.mark.skipif(
        not hardware_info["platforms"]["webgpu"]["available"],
        reason="WebGPU not available",
    )(func)


def skip_if_no_webnn(func):
    import pytest

    hardware_info = detect_hardware()
    return pytest.mark.skipif(
        not hardware_info["platforms"]["webnn"]["available"],
        reason="WebNN not available",
    )(func)


HARDWARE_INFO = detect_hardware()

if __name__ == "__main__":
    import json

    print(json.dumps(detect_hardware(), indent=2))
