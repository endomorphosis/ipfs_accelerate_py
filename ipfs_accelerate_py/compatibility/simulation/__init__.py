"""Explicit simulation namespace. Results here are never live."""

from __future__ import annotations

from .fabricated_hardware import (
    CUDA_MOCK_NAMESPACE,
    MOCK_HARDWARE_NAMESPACE,
    FabricatedHardwareError,
    MockHardwareDetection,
    UnavailableHardwareDetection,
    UnavailableTorch,
    create_simulated_cuda_implementation,
    instantiate_mock_hardware_detection,
    load_ordinary_hardware_detection,
    torch_is_fabricated,
)
from .legacy_mock_coordinator import (
    LEGACY_MOCK_COORDINATOR_NAMESPACE,
    LegacyMockCoordinatorError,
    MockWorker,
    UnavailableWorker,
    admit_legacy_workflow_coordinator,
    instantiate_mock_worker,
    load_ordinary_runtime_worker,
    quarantined_coordinator_unavailable_result,
)

__all__ = (
    "CUDA_MOCK_NAMESPACE",
    "LEGACY_MOCK_COORDINATOR_NAMESPACE",
    "MOCK_HARDWARE_NAMESPACE",
    "FabricatedHardwareError",
    "LegacyMockCoordinatorError",
    "MockHardwareDetection",
    "MockWorker",
    "UnavailableHardwareDetection",
    "UnavailableTorch",
    "UnavailableWorker",
    "admit_legacy_workflow_coordinator",
    "create_simulated_cuda_implementation",
    "instantiate_mock_hardware_detection",
    "instantiate_mock_worker",
    "load_ordinary_hardware_detection",
    "load_ordinary_runtime_worker",
    "quarantined_coordinator_unavailable_result",
    "torch_is_fabricated",
)
