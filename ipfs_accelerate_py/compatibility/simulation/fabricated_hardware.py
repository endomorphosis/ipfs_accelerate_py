"""PCPR-031 quarantined fabricated hardware capability.

Ordinary runtime cannot treat mock hardware detection or CUDA mock
implementations as live availability. Callers must select this
compatibility/simulation namespace explicitly. Simulated hardware is never
live. Missing backends stay typed unavailable.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import ModuleType, SimpleNamespace
from typing import Any, Final
from unittest.mock import MagicMock

from ipfs_accelerate_py.assurance.capability_outcomes import (
    is_explicit_test_mode,
    select_simulation_namespace,
)
from ipfs_accelerate_py.assurance.hardware_capability_ladder import (
    simulated_hardware_map,
    unavailable_hardware_map,
)

MOCK_HARDWARE_NAMESPACE: Final = "mock_hardware"
CUDA_MOCK_NAMESPACE: Final = "cuda_mock_implementation"
ORDINARY_RUNTIME: Final = False
SIMULATED_RESULTS_ARE_LIVE: Final = False


class FabricatedHardwareError(RuntimeError):
    """Ordinary runtime attempted to instantiate fabricated hardware."""

    def __init__(self, message: str, *, code: str = "fabricated_hardware_quarantined"):
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
            raise FabricatedHardwareError(selected.message, code=str(selected.code))
        return
    refused = select_simulation_namespace(
        namespace,
        explicit_test_mode=False,
        environ=environ,
    )
    raise FabricatedHardwareError(
        f"{label} is quarantined behind the compatibility/simulation namespace; "
        "ordinary runtime cannot instantiate it. Pass explicit_simulation=True "
        "or set IPFS_ACCELERATE_EXPLICIT_TEST_MODE. "
        f"{refused.message}",
        code=str(refused.code),
    )


class UnavailableHardwareDetection:
    """Ordinary-runtime stand-in. Hardware flags stay typed unavailable."""

    origin = "unavailable"
    live = False
    simulated = False
    production_authorized = False
    outcome = "Unavailable"

    def detect_all_hardware(self) -> dict[str, Any]:
        return unavailable_hardware_map()

    def detect_hardware(self) -> dict[str, Any]:
        return self.detect_all_hardware()


class MockHardwareDetection:
    """Compatibility/simulation hardware detection. Not live."""

    origin = "simulated"
    live = False
    simulated = True
    production_authorized = False
    outcome = "Simulated"

    def detect_all_hardware(self) -> dict[str, Any]:
        return simulated_hardware_map()

    def detect_hardware(self) -> dict[str, Any]:
        return self.detect_all_hardware()


def load_ordinary_hardware_detection() -> UnavailableHardwareDetection:
    """Ordinary runtime uses typed unavailable hardware, never a mock probe."""

    return UnavailableHardwareDetection()


def instantiate_mock_hardware_detection(
    *,
    explicit_simulation: bool = False,
    explicit_test_mode: bool | None = None,
    environ: Mapping[str, str] | None = None,
) -> MockHardwareDetection:
    """Construct mock hardware detection only under explicit simulation."""

    _require_simulation(
        MOCK_HARDWARE_NAMESPACE,
        explicit_simulation=explicit_simulation,
        explicit_test_mode=explicit_test_mode,
        environ=environ,
        label="Mock hardware detection",
    )
    return MockHardwareDetection()


def mock_hardware_detection_module(
    *,
    explicit_simulation: bool = False,
    explicit_test_mode: bool | None = None,
    environ: Mapping[str, str] | None = None,
) -> ModuleType:
    """Legacy ModuleType surface, gated and labeled Simulated."""

    detector = instantiate_mock_hardware_detection(
        explicit_simulation=explicit_simulation,
        explicit_test_mode=explicit_test_mode,
        environ=environ,
    )
    module = ModuleType("mock_hardware_detection")
    module.detect_all_hardware = detector.detect_all_hardware
    module.detect_hardware = detector.detect_hardware
    module.origin = "simulated"
    module.live = False
    module.simulated = True
    module.production_authorized = False
    module.outcome = "Simulated"
    return module


class UnavailableCuda:
    """Typed-unavailable CUDA adapter. ``is_available`` is null, not False."""

    origin = "unavailable"
    live = False
    production_authorized = False

    def is_available(self) -> None:
        return None

    def device_count(self) -> None:
        return None

    def current_device(self) -> None:
        return None

    def get_device_name(self, *args: object, **kwargs: object) -> None:
        return None

    def empty_cache(self) -> None:
        return None

    def __bool__(self) -> bool:
        return False


class UnavailableTorch:
    """Stand-in when PyTorch is absent. Not a MagicMock and not live CUDA."""

    origin = "unavailable"
    live = False
    production_authorized = False
    cuda = UnavailableCuda()

    def tensor(self, *args: object, **kwargs: object) -> None:
        return None

    def zeros(self, *args: object, **kwargs: object) -> None:
        return None

    def device(self, *args: object, **kwargs: object) -> None:
        return None

    def __bool__(self) -> bool:
        return False


def torch_is_fabricated(torch_obj: Any) -> bool:
    """True when a torch stand-in cannot prove live CUDA."""

    if torch_obj is None:
        return True
    origin = getattr(torch_obj, "origin", None)
    if origin in {"unavailable", "simulated", "fixture"}:
        return True
    name = type(torch_obj).__name__
    if name in {"MagicMock", "Mock", "AsyncMock", "UnavailableTorch"}:
        return True
    module_name = getattr(type(torch_obj), "__module__", "")
    if module_name.startswith("unittest.mock"):
        return True
    return False


def create_simulated_cuda_implementation(
    model_type: str,
    shape_info: Any = None,
    *,
    torch_module: Any = None,
    explicit_simulation: bool = False,
    explicit_test_mode: bool | None = None,
    environ: Mapping[str, str] | None = None,
) -> tuple[Any, Any, Any, Any, int]:
    """CUDA mock implementation. Labeled Simulated. ``is_available`` is not live."""

    _require_simulation(
        CUDA_MOCK_NAMESPACE,
        explicit_simulation=explicit_simulation,
        explicit_test_mode=explicit_test_mode,
        environ=environ,
        label="CUDA mock implementation",
    )
    torch_ref = torch_module if torch_module is not None else UnavailableTorch()
    mock_device = SimpleNamespace(
        type="cuda",
        index=0,
        origin="simulated",
        live=False,
        production_authorized=False,
    )
    cuda_functions = {
        "is_available": MagicMock(return_value=False),
        "get_device_name": MagicMock(return_value="Simulated CUDA Device"),
        "device_count": MagicMock(return_value=0),
        "current_device": MagicMock(return_value=0),
        "empty_cache": MagicMock(),
        "origin": "simulated",
        "live": False,
        "simulated_available": True,
        "production_authorized": False,
        "outcome": "Simulated",
    }
    del cuda_functions

    def _label(payload: dict[str, Any]) -> dict[str, Any]:
        payload["implementation_type"] = "MOCK"
        payload["origin"] = "simulated"
        payload["live"] = False
        payload["outcome"] = "Simulated"
        payload["production_authorized"] = False
        payload["device"] = "simulated-cuda"
        return payload

    if model_type == "lm":
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.half.return_value = mock_model
        mock_model.eval.return_value = mock_model

        def handler(prompt, max_new_tokens=100, temperature=0.7):
            del max_new_tokens, temperature
            return _label(
                {
                    "text": f"(MOCK CUDA) Generated text for: {str(prompt)[:20]}...",
                }
            )

        return None, mock_model, handler, None, 8

    if model_type == "embed":
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.half.return_value = mock_model
        mock_model.eval.return_value = mock_model

        def handler(text):
            del text
            embed_dim = shape_info or 768
            embedding = None
            if not torch_is_fabricated(torch_ref) and hasattr(torch_ref, "zeros"):
                embedding = torch_ref.zeros(embed_dim)
            return _label({"embedding": embedding})

        return None, mock_model, handler, None, 16

    if model_type in {"clip", "xclip"}:

        def handler(text=None, image=None):
            del text, image
            return _label(
                {
                    "text_embedding": None,
                    "image_embedding": None,
                    "similarity": None,
                }
            )

        return None, MagicMock(), handler, None, 8

    return (
        None,
        MagicMock(),
        lambda x: _label({"output": "(MOCK CUDA) Output"}),
        None,
        8,
    )


__all__ = (
    "CUDA_MOCK_NAMESPACE",
    "MOCK_HARDWARE_NAMESPACE",
    "ORDINARY_RUNTIME",
    "SIMULATED_RESULTS_ARE_LIVE",
    "FabricatedHardwareError",
    "MockHardwareDetection",
    "UnavailableCuda",
    "UnavailableHardwareDetection",
    "UnavailableTorch",
    "create_simulated_cuda_implementation",
    "instantiate_mock_hardware_detection",
    "load_ordinary_hardware_detection",
    "mock_hardware_detection_module",
    "torch_is_fabricated",
)
