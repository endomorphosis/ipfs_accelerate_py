"""Native hardware-tools category implementations for unified mcp_server."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _load_hardware_tools_api() -> Dict[str, Any]:
    """Resolve source hardware-tools APIs with compatibility fallback."""
    try:
        from ipfs_accelerate_py.mcp.tools.hardware import (  # type: ignore
            get_hardware_info as _get_hardware_info,
            get_basic_hardware_info as _get_basic_hardware_info,
            test_hardware as _test_hardware,
            perform_basic_hardware_test as _perform_basic_hardware_test,
            recommend_hardware as _recommend_hardware,
            get_basic_hardware_recommendations as _get_basic_hardware_recommendations,
        )

        return {
            "get_hardware_info": _get_hardware_info,
            "get_basic_hardware_info": _get_basic_hardware_info,
            "test_hardware": _test_hardware,
            "perform_basic_hardware_test": _perform_basic_hardware_test,
            "recommend_hardware": _recommend_hardware,
            "get_basic_hardware_recommendations": _get_basic_hardware_recommendations,
        }
    except Exception:
        logger.warning(
            "Source hardware_tools import unavailable, using typed-unavailable fallback"
        )
        from ipfs_accelerate_py.compatibility.simulation.fabricated_endpoint_success import (
            unavailable_hardware_info,
            unavailable_hardware_recommendation,
            unavailable_hardware_test,
        )

        def _hardware_info_fallback(include_detailed: bool = False) -> Dict[str, Any]:
            return unavailable_hardware_info(include_detailed=include_detailed)

        def _test_hardware_fallback(
            accelerator: str = "all", test_level: str = "basic"
        ) -> Dict[str, Any]:
            return unavailable_hardware_test(accelerator, test_level)

        def _recommend_hardware_fallback(
            model_type: str = "general",
            model_size: str = "medium",
            task: str = "inference",
        ) -> Dict[str, Any]:
            return unavailable_hardware_recommendation(model_type, model_size, task)

        return {
            "get_hardware_info": _hardware_info_fallback,
            "get_basic_hardware_info": _hardware_info_fallback,
            "test_hardware": _test_hardware_fallback,
            "perform_basic_hardware_test": _test_hardware_fallback,
            "recommend_hardware": _recommend_hardware_fallback,
            "get_basic_hardware_recommendations": _recommend_hardware_fallback,
        }


_API = _load_hardware_tools_api()


def _normalize_payload(payload: Any) -> Dict[str, Any]:
    """Normalize delegate payloads. Missing results stay typed unavailable."""
    if isinstance(payload, dict):
        envelope = dict(payload)
        outcome = envelope.get("outcome")
        if outcome in {"Unavailable", "Rejected", "Failed", "Simulated", "Unknown"}:
            envelope["live"] = False
            envelope.setdefault("status", str(outcome).lower())
            return envelope
        failed = bool(envelope.get("error")) or envelope.get("success") is False
        if failed:
            envelope["status"] = "error"
            envelope.setdefault("outcome", "Failed")
            envelope["live"] = False
        elif "status" not in envelope:
            if envelope.get("live") is True:
                envelope["status"] = "success"
            else:
                envelope["status"] = "unavailable"
                envelope.setdefault("outcome", "Unavailable")
                envelope["live"] = False
        return envelope
    if payload is None:
        return {
            "status": "unavailable",
            "outcome": "Unavailable",
            "live": False,
            "code": "hardware_result_absent",
        }
    return {
        "status": "unavailable",
        "outcome": "Unavailable",
        "live": False,
        "code": "hardware_result_untyped",
        "result": payload,
    }


def _error_result(message: str, **context: Any) -> Dict[str, Any]:
    """Build consistent error envelope for wrapper edge failures."""
    envelope: Dict[str, Any] = {
        "status": "error",
        "success": False,
        "error": message,
    }
    envelope.update(context)
    return envelope


async def hardware_get_info(include_detailed: bool = False) -> Dict[str, Any]:
    """Return information about available hardware accelerators."""
    try:
        result = _API["get_hardware_info"](include_detailed=include_detailed)
        return _normalize_payload(result)
    except Exception as exc:
        return _error_result(str(exc), include_detailed=include_detailed)


async def hardware_get_basic_info(include_detailed: bool = False) -> Dict[str, Any]:
    """Return basic hardware information without requiring IPFS Accelerate."""
    try:
        result = _API["get_basic_hardware_info"](include_detailed=include_detailed)
        return _normalize_payload(result)
    except Exception as exc:
        return _error_result(str(exc), include_detailed=include_detailed)


async def hardware_test(accelerator: str = "all", test_level: str = "basic") -> Dict[str, Any]:
    """Run hardware tests on the specified accelerator."""
    try:
        result = _API["test_hardware"](accelerator=accelerator, test_level=test_level)
        return _normalize_payload(result)
    except Exception as exc:
        return _error_result(str(exc), accelerator=accelerator, test_level=test_level)


async def hardware_recommend(
    model_type: str = "general",
    model_size: str = "medium",
    task: str = "inference",
) -> Dict[str, Any]:
    """Recommend hardware configuration for a given model type and task."""
    try:
        result = _API["recommend_hardware"](model_type=model_type, model_size=model_size, task=task)
        return _normalize_payload(result)
    except Exception as exc:
        return _error_result(str(exc), model_type=model_type, model_size=model_size, task=task)


def register_native_hardware_tools(manager: Any) -> None:
    """Register native hardware-tools category tools in unified manager."""
    manager.register_tool(
        category="hardware_tools",
        name="hardware_get_info",
        func=hardware_get_info,
        description="Get information about available hardware accelerators (CPU, GPU, MPS, OpenVINO).",
        input_schema={
            "type": "object",
            "properties": {
                "include_detailed": {
                    "type": "boolean",
                    "description": "Include detailed hardware information.",
                    "default": False,
                }
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "hardware-tools"],
    )
    manager.register_tool(
        category="hardware_tools",
        name="hardware_get_basic_info",
        func=hardware_get_basic_info,
        description="Get basic hardware information without requiring IPFS Accelerate.",
        input_schema={
            "type": "object",
            "properties": {
                "include_detailed": {
                    "type": "boolean",
                    "description": "Include detailed hardware information.",
                    "default": False,
                }
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "hardware-tools"],
    )
    manager.register_tool(
        category="hardware_tools",
        name="hardware_test",
        func=hardware_test,
        description="Run hardware tests on a specified accelerator.",
        input_schema={
            "type": "object",
            "properties": {
                "accelerator": {
                    "type": "string",
                    "description": "Accelerator to test: 'all', 'cpu', 'cuda', 'mps', 'openvino'.",
                    "default": "all",
                },
                "test_level": {
                    "type": "string",
                    "description": "Test depth: 'basic' or 'full'.",
                    "default": "basic",
                },
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "hardware-tools"],
    )
    manager.register_tool(
        category="hardware_tools",
        name="hardware_recommend",
        func=hardware_recommend,
        description="Recommend hardware configuration for a model type and task.",
        input_schema={
            "type": "object",
            "properties": {
                "model_type": {
                    "type": "string",
                    "description": "Type of model (e.g., 'text-generation', 'image-classification').",
                    "default": "general",
                },
                "model_size": {
                    "type": "string",
                    "description": "Size of model: 'small', 'medium', or 'large'.",
                    "default": "medium",
                },
                "task": {
                    "type": "string",
                    "description": "Task type: 'inference' or 'training'.",
                    "default": "inference",
                },
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "hardware-tools"],
    )
