"""Native acceleration-tools category implementations for unified mcp_server.

Exposes IPFS hardware acceleration operations (detect, accelerate, benchmark,
status) from the legacy ``ipfs_accelerate_py.mcp.tools.acceleration`` module
through the unified MCP++ tool dispatch surface.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _load_acceleration_tools_api() -> Dict[str, Any]:
    """Resolve source acceleration-tools APIs with compatibility fallback."""
    try:
        # Import the standalone non-decorator functions from the legacy module.
        from ipfs_accelerate_py.mcp.tools.acceleration import (  # type: ignore
            register_acceleration_tools as _register,
        )
        # The inner tool callables are nested inside register_acceleration_tools,
        # so we resolve the underlying backends directly.
    except Exception:
        pass

    # Resolve directly from backends used by the legacy module.
    try:
        from ipfs_accelerate_py import hardware_detection as _hw  # type: ignore

        def _get_hw_info(ctx: Any = None) -> Dict[str, Any]:
            return _hw.get_hardware_info()

        return {"get_hardware_info": _get_hw_info}
    except Exception:
        pass

    logger.warning(
        "Source acceleration_tools import unavailable, using fallback stubs"
    )

    def _hw_info_fallback(ctx: Any = None) -> Dict[str, Any]:
        import platform
        return {
            "status": "success",
            "platform": platform.system(),
            "cpu": {"available": True, "name": platform.processor()},
            "cuda": {"available": False},
            "mps": {"available": False},
            "openvino": {"available": False},
        }

    return {"get_hardware_info": _hw_info_fallback}


_API = _load_acceleration_tools_api()


def _normalize_payload(payload: Any) -> Dict[str, Any]:
    """Normalize delegate payloads to deterministic dict envelopes."""
    if isinstance(payload, dict):
        envelope = dict(payload)
        failed = bool(envelope.get("error")) or envelope.get("success") is False
        if failed:
            envelope["status"] = "error"
        elif "status" not in envelope:
            envelope["status"] = "success"
        return envelope
    if payload is None:
        return {"status": "success"}
    return {"status": "success", "result": payload}


def _error_result(message: str, **context: Any) -> Dict[str, Any]:
    """Build consistent error envelope for wrapper edge failures."""
    envelope: Dict[str, Any] = {
        "status": "error",
        "success": False,
        "error": message,
    }
    envelope.update(context)
    return envelope


async def acceleration_get_hardware_info() -> Dict[str, Any]:
    """Return hardware acceleration capabilities detected on this host."""
    try:
        fn = _API.get("get_hardware_info")
        if fn is not None:
            result = fn()
            return _normalize_payload(result)
        return _normalize_payload({"hardware": {}, "detected": False})
    except Exception as exc:
        return _error_result(str(exc))


async def acceleration_accelerate_model(
    cid: str,
    device: str = "auto",
) -> Dict[str, Any]:
    """Load and accelerate a model identified by IPFS CID onto a device."""
    try:
        try:
            from ipfs_accelerate_py.ipfs_accelerate import ipfs_accelerate_py  # type: ignore

            instance = ipfs_accelerate_py()
            result = await instance.accelerate(cid=cid, device=device)
            return _normalize_payload(result if isinstance(result, dict) else {"cid": cid, "device": device, "accelerated": result})
        except Exception:
            pass
        return _normalize_payload({"cid": cid, "device": device, "accelerated": False, "backend_available": False})
    except Exception as exc:
        return _error_result(str(exc), cid=cid, device=device)


async def acceleration_benchmark_model(
    cid: str,
    device: str = "auto",
    iterations: int = 5,
) -> Dict[str, Any]:
    """Benchmark a model identified by IPFS CID on a device."""
    try:
        try:
            from ipfs_accelerate_py.ipfs_accelerate import ipfs_accelerate_py  # type: ignore

            instance = ipfs_accelerate_py()
            result = await instance.benchmark(cid=cid, device=device, iterations=iterations)
            return _normalize_payload(result if isinstance(result, dict) else {"cid": cid, "device": device, "iterations": iterations, "benchmark": result})
        except Exception:
            pass
        return _normalize_payload(
            {
                "cid": cid,
                "device": device,
                "iterations": iterations,
                "benchmark": {},
                "backend_available": False,
            }
        )
    except Exception as exc:
        return _error_result(str(exc), cid=cid, device=device)


async def acceleration_model_status(cid: str) -> Dict[str, Any]:
    """Get acceleration status for a model identified by IPFS CID."""
    try:
        try:
            from ipfs_accelerate_py.ipfs_accelerate import ipfs_accelerate_py  # type: ignore

            instance = ipfs_accelerate_py()
            result = await instance.status(cid=cid)
            return _normalize_payload(result if isinstance(result, dict) else {"cid": cid, "status_data": result})
        except Exception:
            pass
        return _normalize_payload({"cid": cid, "loaded": False, "backend_available": False})
    except Exception as exc:
        return _error_result(str(exc), cid=cid)


def register_native_acceleration_tools(manager: Any) -> None:
    """Register native acceleration-tools category tools in unified manager."""
    manager.register_tool(
        category="acceleration_tools",
        name="acceleration_get_hardware_info",
        func=acceleration_get_hardware_info,
        description="Return hardware acceleration capabilities detected on this host.",
        input_schema={"type": "object", "properties": {}, "required": []},
        runtime="fastapi",
        tags=["native", "mcpp", "acceleration-tools"],
    )
    manager.register_tool(
        category="acceleration_tools",
        name="acceleration_accelerate_model",
        func=acceleration_accelerate_model,
        description="Load and accelerate a model identified by IPFS CID onto a device.",
        input_schema={
            "type": "object",
            "properties": {
                "cid": {"type": "string", "description": "IPFS CID of the model."},
                "device": {
                    "type": "string",
                    "description": "Target device: 'auto', 'cpu', 'cuda', 'mps'.",
                    "default": "auto",
                },
            },
            "required": ["cid"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "acceleration-tools"],
    )
    manager.register_tool(
        category="acceleration_tools",
        name="acceleration_benchmark_model",
        func=acceleration_benchmark_model,
        description="Benchmark a model identified by IPFS CID on a device.",
        input_schema={
            "type": "object",
            "properties": {
                "cid": {"type": "string", "description": "IPFS CID of the model."},
                "device": {
                    "type": "string",
                    "description": "Target device.",
                    "default": "auto",
                },
                "iterations": {
                    "type": "integer",
                    "description": "Number of benchmark iterations.",
                    "default": 5,
                },
            },
            "required": ["cid"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "acceleration-tools"],
    )
    manager.register_tool(
        category="acceleration_tools",
        name="acceleration_model_status",
        func=acceleration_model_status,
        description="Get acceleration status for a model identified by IPFS CID.",
        input_schema={
            "type": "object",
            "properties": {
                "cid": {"type": "string", "description": "IPFS CID of the model."}
            },
            "required": ["cid"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "acceleration-tools"],
    )
