"""Explicit simulation namespace. Results here are never live."""

from __future__ import annotations

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
    "LEGACY_MOCK_COORDINATOR_NAMESPACE",
    "LegacyMockCoordinatorError",
    "MockWorker",
    "UnavailableWorker",
    "admit_legacy_workflow_coordinator",
    "instantiate_mock_worker",
    "load_ordinary_runtime_worker",
    "quarantined_coordinator_unavailable_result",
)
