"""PCPR-030 quarantined Accelerate legacy mock coordinator.

Ordinary runtime constructors cannot instantiate ``MockWorker`` or the
fallback-without-backend workflow coordinator. Callers must select this
compatibility/simulation namespace explicitly. Simulated hardware,
providers, and queues are never live. Missing workers stay typed
unavailable.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from ipfs_accelerate_py.assurance.capability_outcomes import (
    is_explicit_test_mode,
    select_simulation_namespace,
)

LEGACY_MOCK_COORDINATOR_NAMESPACE: Final = "legacy_mock_coordinator"
MOCK_WORKER_NAMESPACE: Final = "mock_worker"
ORDINARY_RUNTIME: Final = False
SIMULATED_RESULTS_ARE_LIVE: Final = False


class LegacyMockCoordinatorError(RuntimeError):
    """Ordinary runtime attempted to instantiate a quarantined mock coordinator."""

    def __init__(self, message: str, *, code: str = "legacy_mock_coordinator_quarantined"):
        super().__init__(message)
        self.code = code
        self.outcome = "Unavailable"
        self.live = False
        self.simulated = False


class MockWorker:
    """Compatibility/simulation worker. Not a live backend and not ordinary runtime."""

    origin = "simulated"
    live = False
    simulated = True
    production_authorized = False

    def init_worker(self, *args: object, **kwargs: object) -> dict[str, Any]:
        return {
            "origin": "simulated",
            "live": False,
            "outcome": "Simulated",
            "code": "mock_worker_simulation_only",
        }

    def test_hardware(self, *args: object, **kwargs: object) -> dict[str, Any]:
        return {
            "cuda": None,
            "openvino": None,
            "llama_cpp": None,
            "ipex": None,
            "qualcomm": None,
            "apple": None,
            "webnn": None,
            "origin": "simulated",
            "live": False,
            "outcome": "Simulated",
            "code": "mock_worker_simulation_only",
            "production_authorized": False,
        }


class UnavailableWorker:
    """Ordinary-runtime stand-in when no live worker is admitted.

    Hardware flags stay typed unavailable (null), never True and never a
    fabricated False that could be read as a measured probe.
    """

    origin = "unavailable"
    live = False
    simulated = False
    production_authorized = False

    async def init_worker(self, *args: object, **kwargs: object) -> dict[str, Any]:
        return {
            "origin": "unavailable",
            "live": False,
            "outcome": "Unavailable",
            "code": "worker_unavailable",
        }

    def test_hardware(self, *args: object, **kwargs: object) -> dict[str, Any]:
        return {
            "cuda": None,
            "openvino": None,
            "llama_cpp": None,
            "ipex": None,
            "qualcomm": None,
            "apple": None,
            "webnn": None,
            "origin": "unavailable",
            "live": False,
            "outcome": "Unavailable",
            "code": "worker_unavailable",
            "production_authorized": False,
        }


def _explicit_simulation_requested(
    *,
    explicit_simulation: bool,
    config: Mapping[str, Any] | None,
    explicit_test_mode: bool | None,
    environ: Mapping[str, str] | None,
) -> bool:
    if explicit_simulation is True:
        return True
    if config is not None:
        for key in ("explicit_simulation", "simulation", "simulation_only"):
            if bool(config.get(key)):
                return True
    return is_explicit_test_mode(explicit_test_mode=explicit_test_mode, environ=environ)


def admit_legacy_workflow_coordinator(
    *,
    explicit_simulation: bool = False,
    config: Mapping[str, Any] | None = None,
    explicit_test_mode: bool | None = None,
    environ: Mapping[str, str] | None = None,
) -> None:
    """Allow WorkflowCoordinator only under explicit simulation/test selection."""

    if _explicit_simulation_requested(
        explicit_simulation=explicit_simulation,
        config=config,
        explicit_test_mode=explicit_test_mode,
        environ=environ,
    ):
        selected = select_simulation_namespace(
            LEGACY_MOCK_COORDINATOR_NAMESPACE,
            explicit_test_mode=True,
            environ=environ,
        )
        if selected.outcome != "Simulated":
            raise LegacyMockCoordinatorError(
                selected.message,
                code=str(selected.code),
            )
        return
    refused = select_simulation_namespace(
        LEGACY_MOCK_COORDINATOR_NAMESPACE,
        explicit_test_mode=False,
        environ=environ,
    )
    raise LegacyMockCoordinatorError(
        "WorkflowCoordinator is quarantined behind the compatibility/"
        "simulation namespace; ordinary runtime cannot instantiate it. "
        "Pass explicit_simulation=True or set IPFS_ACCELERATE_EXPLICIT_TEST_MODE. "
        f"{refused.message}",
        code=str(refused.code),
    )


def instantiate_mock_worker(
    *,
    explicit_simulation: bool = False,
    explicit_test_mode: bool | None = None,
    environ: Mapping[str, str] | None = None,
) -> MockWorker:
    """Construct MockWorker only under explicit simulation/test selection."""

    if not _explicit_simulation_requested(
        explicit_simulation=explicit_simulation,
        config=None,
        explicit_test_mode=explicit_test_mode,
        environ=environ,
    ):
        refused = select_simulation_namespace(
            MOCK_WORKER_NAMESPACE,
            explicit_test_mode=False,
            environ=environ,
        )
        raise LegacyMockCoordinatorError(
            "MockWorker is quarantined behind the compatibility/simulation "
            "namespace; ordinary runtime cannot instantiate it. "
            f"{refused.message}",
            code=str(refused.code),
        )
    selected = select_simulation_namespace(
        MOCK_WORKER_NAMESPACE,
        explicit_test_mode=True,
        environ=environ,
    )
    if selected.outcome != "Simulated":
        raise LegacyMockCoordinatorError(selected.message, code=str(selected.code))
    return MockWorker()


def load_ordinary_runtime_worker(
    resources: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> object:
    """Return a non-mock worker for ordinary runtime constructors.

    Injected ``resources['worker']`` values named MockWorker are rejected.
    Live worker import is not performed here: CPU/CUDA qualification is a
    later PCPR task. Missing live workers stay typed unavailable.
    """

    del metadata
    if resources is not None and "worker" in resources and resources["worker"] is not None:
        worker = resources["worker"]
        if type(worker).__name__ == "MockWorker" or isinstance(worker, MockWorker):
            raise LegacyMockCoordinatorError(
                "ordinary runtime cannot install MockWorker as the live worker"
            )
        return worker
    return UnavailableWorker()


def quarantined_coordinator_unavailable_result(
    *,
    task_id: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Typed unavailable envelope for production MCP/runtime callers."""

    payload: dict[str, Any] = {
        "status": "unavailable",
        "success": False,
        "outcome": "Unavailable",
        "code": "legacy_mock_coordinator_quarantined",
        "live": False,
        "simulated": False,
        "error": error
        or (
            "legacy mock coordinator is quarantined; ordinary runtime cannot "
            "instantiate it"
        ),
        "mediation": "mcp_plus_plus",
        "package_id": "ipfs_accelerate_py",
    }
    if task_id is not None:
        payload["task_id"] = task_id
    return payload


__all__ = (
    "LEGACY_MOCK_COORDINATOR_NAMESPACE",
    "MOCK_WORKER_NAMESPACE",
    "ORDINARY_RUNTIME",
    "SIMULATED_RESULTS_ARE_LIVE",
    "LegacyMockCoordinatorError",
    "MockWorker",
    "UnavailableWorker",
    "admit_legacy_workflow_coordinator",
    "instantiate_mock_worker",
    "load_ordinary_runtime_worker",
    "quarantined_coordinator_unavailable_result",
)
