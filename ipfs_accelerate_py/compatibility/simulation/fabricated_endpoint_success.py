"""PCPR-033 quarantined fabricated endpoint success.

Ordinary runtime cannot treat mock endpoint handlers, MCP mock inference, or
hardware-test fallbacks as live success. Callers must select this
compatibility/simulation namespace explicitly. Simulated endpoints are never
live. Missing backends, models, canaries, and probes stay typed unavailable.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from ipfs_accelerate_py.assurance.capability_outcomes import (
    is_explicit_test_mode,
    resolve_inference_outcome,
    select_simulation_namespace,
)

MOCK_HANDLER_NAMESPACE: Final = "mock_handler"
MOCK_INFERENCE_NAMESPACE: Final = "mock_inference"
HARDWARE_TEST_FALLBACK_NAMESPACE: Final = "hardware_test_fallback"
ORDINARY_RUNTIME: Final = False
SIMULATED_RESULTS_ARE_LIVE: Final = False
LIVE_ENDPOINT_REQUIRES: Final = (
    "backend",
    "model",
    "canary",
    "timeout",
    "cancellation",
    "resource_admission",
)


class FabricatedEndpointSuccessError(RuntimeError):
    """Ordinary runtime attempted to treat a mock endpoint as live success."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "fabricated_endpoint_success_quarantined",
    ):
        super().__init__(message)
        self.code = code
        self.outcome = "Unavailable"
        self.live = False
        self.simulated = False
        self.production_authorized = False


def _explicit_simulation_requested(
    *,
    explicit_simulation: bool,
    explicit_test_mode: bool | None,
    environ: Mapping[str, str] | None,
) -> bool:
    if explicit_simulation is True:
        return True
    return is_explicit_test_mode(explicit_test_mode=explicit_test_mode, environ=environ)


def _require_simulation(
    namespace: str,
    *,
    explicit_simulation: bool,
    explicit_test_mode: bool | None,
    environ: Mapping[str, str] | None,
    label: str,
) -> None:
    if _explicit_simulation_requested(
        explicit_simulation=explicit_simulation,
        explicit_test_mode=explicit_test_mode,
        environ=environ,
    ):
        selected = select_simulation_namespace(
            namespace,
            explicit_test_mode=True,
            environ=environ,
        )
        if selected.outcome != "Simulated":
            raise FabricatedEndpointSuccessError(selected.message, code=str(selected.code))
        return
    refused = select_simulation_namespace(
        namespace,
        explicit_test_mode=False,
        environ=environ,
    )
    raise FabricatedEndpointSuccessError(
        f"{label} is quarantined behind the compatibility/simulation namespace; "
        "ordinary runtime cannot instantiate it. Pass explicit_simulation=True "
        "or set IPFS_ACCELERATE_EXPLICIT_TEST_MODE. "
        f"{refused.message}",
        code=str(refused.code),
    )


def unavailable_endpoint_envelope(
    *,
    model: str,
    endpoint_type: str,
    code: str = "endpoint_success_requires_live_evidence",
) -> dict[str, Any]:
    """Typed unavailable endpoint result. Never status=success."""

    return {
        "status": "unavailable",
        "outcome": "Unavailable",
        "origin": "unavailable",
        "live": False,
        "simulated": False,
        "production_authorized": False,
        "implementation_type": "UNAVAILABLE",
        "model": model,
        "endpoint": endpoint_type,
        "code": code,
        "required_live_evidence": list(LIVE_ENDPOINT_REQUIRES),
    }


def simulated_endpoint_envelope(
    *,
    model: str,
    endpoint_type: str,
    payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Explicit simulation envelope. Never live and never REAL."""

    return {
        "status": "simulated",
        "outcome": "Simulated",
        "origin": "simulated",
        "live": False,
        "simulated": True,
        "production_authorized": False,
        "implementation_type": "MOCK",
        "model": model,
        "endpoint": endpoint_type,
        "code": "endpoint_handler_simulated_test_mode",
        "payload": dict(payload or {}),
    }


class UnavailableEndpointHandler:
    """Ordinary-runtime stand-in. Registered endpoints stay typed unavailable."""

    origin = "unavailable"
    live = False
    simulated = False
    production_authorized = False
    outcome = "Unavailable"
    implementation_type = "UNAVAILABLE"

    def __init__(self, model: str, endpoint_type: str):
        self.model = model
        self.endpoint_type = endpoint_type

    def __call__(self, input_data: Any = None) -> dict[str, Any]:
        envelope = unavailable_endpoint_envelope(
            model=self.model,
            endpoint_type=self.endpoint_type,
        )
        envelope["input_present"] = input_data is not None
        return envelope

    async def __call_async__(self, input_data: Any = None) -> dict[str, Any]:
        return self(input_data)


class SimulatedEndpointHandler:
    """Compatibility/simulation endpoint handler. Not live."""

    origin = "simulated"
    live = False
    simulated = True
    production_authorized = False
    outcome = "Simulated"
    implementation_type = "MOCK"

    def __init__(self, model: str, endpoint_type: str):
        self.model = model
        self.endpoint_type = endpoint_type

    def __call__(self, input_data: Any = None) -> dict[str, Any]:
        return simulated_endpoint_envelope(
            model=self.model,
            endpoint_type=self.endpoint_type,
            payload={"input_present": input_data is not None},
        )

    async def __call_async__(self, input_data: Any = None) -> dict[str, Any]:
        return self(input_data)


def create_unavailable_endpoint_handler(
    model: str,
    endpoint_type: str,
) -> UnavailableEndpointHandler:
    """Ordinary runtime installs typed unavailable handlers, never mock success."""

    return UnavailableEndpointHandler(str(model), str(endpoint_type))


def create_simulated_endpoint_handler(
    model: str,
    endpoint_type: str,
    *,
    explicit_simulation: bool = False,
    explicit_test_mode: bool | None = None,
    environ: Mapping[str, str] | None = None,
) -> SimulatedEndpointHandler:
    """Construct a mock endpoint handler only under explicit simulation."""

    _require_simulation(
        MOCK_HANDLER_NAMESPACE,
        explicit_simulation=explicit_simulation,
        explicit_test_mode=explicit_test_mode,
        environ=environ,
        label="Mock endpoint handler",
    )
    return SimulatedEndpointHandler(str(model), str(endpoint_type))


def _ensure_handler_slot(resources: dict[str, Any], model: str) -> dict[str, Any]:
    handlers = resources.setdefault("endpoint_handler", {})
    if model not in handlers or not isinstance(handlers.get(model), dict):
        handlers[model] = {}
    return handlers[model]


def install_ordinary_endpoint_handler(
    resources: dict[str, Any],
    model: str,
    endpoint_type: str,
) -> UnavailableEndpointHandler:
    """Store a typed-unavailable handler. Do not advertise a live endpoint."""

    handler = create_unavailable_endpoint_handler(model, endpoint_type)
    slot = _ensure_handler_slot(resources, model)
    slot[endpoint_type] = handler
    return handler


def install_simulated_endpoint_handler(
    resources: dict[str, Any],
    model: str,
    endpoint_type: str,
    *,
    explicit_simulation: bool = False,
    explicit_test_mode: bool | None = None,
    environ: Mapping[str, str] | None = None,
) -> SimulatedEndpointHandler:
    """Store a Simulated handler only under explicit simulation."""

    handler = create_simulated_endpoint_handler(
        model,
        endpoint_type,
        explicit_simulation=explicit_simulation,
        explicit_test_mode=explicit_test_mode,
        environ=environ,
    )
    slot = _ensure_handler_slot(resources, model)
    slot[endpoint_type] = handler
    return handler


def install_endpoint_handler(
    resources: dict[str, Any],
    model: str,
    endpoint_type: str,
    *,
    explicit_simulation: bool = False,
    explicit_test_mode: bool | None = None,
    environ: Mapping[str, str] | None = None,
) -> UnavailableEndpointHandler | SimulatedEndpointHandler:
    """Ordinary path is unavailable; simulated path requires explicit opt-in."""

    if _explicit_simulation_requested(
        explicit_simulation=explicit_simulation,
        explicit_test_mode=explicit_test_mode,
        environ=environ,
    ):
        return install_simulated_endpoint_handler(
            resources,
            model,
            endpoint_type,
            explicit_simulation=True,
            explicit_test_mode=explicit_test_mode,
            environ=environ,
        )
    return install_ordinary_endpoint_handler(resources, model, endpoint_type)


def ordinary_mock_inference(
    task_type: str,
    model_id: str,
    input_data: Any = None,
) -> dict[str, Any]:
    """Ordinary MCP inference without live evidence is typed unavailable."""

    resolved = resolve_inference_outcome(
        backend="cpu",
        model=model_id,
        simulated=True,
        mock_handler=True,
        explicit_test_mode=False,
        details={"task_type": task_type},
    )
    return {
        "status": "unavailable",
        "outcome": resolved.outcome,
        "origin": "unavailable",
        "live": False,
        "simulated": False,
        "production_authorized": False,
        "implementation_type": "UNAVAILABLE",
        "task_type": task_type,
        "model": model_id,
        "code": resolved.code,
        "message": resolved.message,
        "required_live_evidence": list(LIVE_ENDPOINT_REQUIRES),
        "input_present": input_data is not None,
        "fallback_success_forbidden": True,
    }


def simulate_inference(
    task_type: str,
    model_id: str,
    input_data: Any = None,
    *,
    explicit_simulation: bool = False,
    explicit_test_mode: bool | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Mock inference payload only under explicit simulation. Never live."""

    _require_simulation(
        MOCK_INFERENCE_NAMESPACE,
        explicit_simulation=explicit_simulation,
        explicit_test_mode=explicit_test_mode,
        environ=environ,
        label="Mock inference",
    )
    resolved = resolve_inference_outcome(
        backend="cpu",
        model=model_id,
        simulated=True,
        mock_handler=True,
        explicit_test_mode=True,
        details={"task_type": task_type},
    )
    return {
        "status": "simulated",
        "outcome": resolved.outcome,
        "origin": "simulated",
        "live": False,
        "simulated": True,
        "production_authorized": False,
        "implementation_type": "MOCK",
        "task_type": task_type,
        "model": model_id,
        "code": resolved.code,
        "message": resolved.message,
        "payload": {"input_present": input_data is not None},
        "fallback_success_forbidden": True,
    }


def run_inference(
    task_type: str,
    model_id: str,
    input_data: Any = None,
    *,
    explicit_simulation: bool = False,
    explicit_test_mode: bool | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Dispatch ordinary unavailable vs explicit simulated inference."""

    if _explicit_simulation_requested(
        explicit_simulation=explicit_simulation,
        explicit_test_mode=explicit_test_mode,
        environ=environ,
    ):
        return simulate_inference(
            task_type,
            model_id,
            input_data,
            explicit_simulation=True,
            explicit_test_mode=explicit_test_mode,
            environ=environ,
        )
    return ordinary_mock_inference(task_type, model_id, input_data)


def unavailable_hardware_test(
    accelerator: str = "all",
    test_level: str = "basic",
) -> dict[str, Any]:
    """Hardware-test fallback without a live probe is typed unavailable."""

    return {
        "status": "unavailable",
        "outcome": "Unavailable",
        "origin": "unavailable",
        "live": False,
        "simulated": False,
        "production_authorized": False,
        "accelerator": accelerator,
        "test_level": test_level,
        "results": {},
        "overall_passed": None,
        "code": "hardware_test_requires_live_probe",
        "fallback_success_forbidden": True,
    }


def simulate_hardware_test(
    accelerator: str = "all",
    test_level: str = "basic",
    *,
    explicit_simulation: bool = False,
    explicit_test_mode: bool | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Simulated hardware test. overall_passed is never live True."""

    _require_simulation(
        HARDWARE_TEST_FALLBACK_NAMESPACE,
        explicit_simulation=explicit_simulation,
        explicit_test_mode=explicit_test_mode,
        environ=environ,
        label="Hardware-test fallback",
    )
    return {
        "status": "simulated",
        "outcome": "Simulated",
        "origin": "simulated",
        "live": False,
        "simulated": True,
        "production_authorized": False,
        "accelerator": accelerator,
        "test_level": test_level,
        "results": {},
        "overall_passed": None,
        "code": "hardware_test_simulated_test_mode",
        "fallback_success_forbidden": True,
    }


def unavailable_hardware_info(*, include_detailed: bool = False) -> dict[str, Any]:
    """Platform strings are not a live hardware probe."""

    return {
        "status": "unavailable",
        "outcome": "Unavailable",
        "origin": "unavailable",
        "live": False,
        "simulated": False,
        "production_authorized": False,
        "include_detailed": include_detailed,
        "cpu": {
            "available": None,
            "origin": "absent",
            "outcome": "Unavailable",
            "live": False,
            "production_authorized": False,
        },
        "cuda": {
            "available": None,
            "origin": "absent",
            "outcome": "Unavailable",
            "live": False,
            "production_authorized": False,
        },
        "mps": {
            "available": None,
            "origin": "absent",
            "outcome": "Unavailable",
            "live": False,
            "production_authorized": False,
        },
        "openvino": {
            "available": None,
            "origin": "absent",
            "outcome": "Unavailable",
            "live": False,
            "production_authorized": False,
        },
        "code": "hardware_info_requires_live_probe",
        "fallback_success_forbidden": True,
    }


def unavailable_hardware_recommendation(
    model_type: str = "general",
    model_size: str = "medium",
    task: str = "inference",
) -> dict[str, Any]:
    """Fallback recommendation is not a live hardware qualification."""

    return {
        "status": "unavailable",
        "outcome": "Unavailable",
        "origin": "unavailable",
        "live": False,
        "simulated": False,
        "production_authorized": False,
        "model_type": model_type,
        "model_size": model_size,
        "task": task,
        "recommendation": None,
        "code": "hardware_recommendation_requires_live_probe",
        "fallback_success_forbidden": True,
    }


def run_hardware_test_fallback(
    accelerator: str = "all",
    test_level: str = "basic",
    *,
    explicit_simulation: bool = False,
    explicit_test_mode: bool | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Ordinary fallback is unavailable; simulation never reports passed live."""

    if _explicit_simulation_requested(
        explicit_simulation=explicit_simulation,
        explicit_test_mode=explicit_test_mode,
        environ=environ,
    ):
        return simulate_hardware_test(
            accelerator,
            test_level,
            explicit_simulation=True,
            explicit_test_mode=explicit_test_mode,
            environ=environ,
        )
    return unavailable_hardware_test(accelerator, test_level)
