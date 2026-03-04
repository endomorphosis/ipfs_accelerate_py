"""Native lizardpersons-function-tools category implementations for unified mcp_server."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple, Union

logger = logging.getLogger(__name__)


def _load_lizardpersons_function_api() -> Dict[str, Any]:
    """Resolve source lizardpersons-function-tools APIs with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.lizardpersons_function_tools.llm_context_tools.get_current_time import (  # type: ignore
            get_current_time as _get_current_time,
        )

        return {"get_current_time": _get_current_time}
    except Exception:
        logger.warning(
            "Source lizardpersons_function_tools import unavailable, using fallback time function"
        )

        def _get_current_time_fallback(
            format_type: str = "iso",
            time_between: Optional[Tuple[Union[str, int, float], ...]] = None,
            deadline_date: Optional[Union[str, int, float]] = None,
            check_if_within_working_hours: bool = False,
        ) -> str:
            _ = (time_between, deadline_date, check_if_within_working_hours)
            if format_type == "timestamp":
                return "0"
            if format_type == "human":
                return "1970-01-01 00:00:00"
            return "1970-01-01T00:00:00"

        return {"get_current_time": _get_current_time_fallback}


_API = _load_lizardpersons_function_api()


def _error_result(message: str, **context: Any) -> Dict[str, Any]:
    """Build consistent validation/error envelope for wrapper edge failures."""
    envelope: Dict[str, Any] = {
        "status": "error",
        "success": False,
        "error": message,
    }
    envelope.update(context)
    return envelope


async def get_current_time(
    format_type: str = "iso",
    check_if_within_working_hours: bool = False,
) -> Dict[str, Any]:
    """Get current time via lizardperson function-tool compatibility shim."""
    allowed_formats = {"iso", "human", "timestamp"}
    if not isinstance(format_type, str) or not format_type.strip():
        return _error_result("format_type must be a non-empty string", format_type=format_type)
    if format_type.strip() not in allowed_formats:
        return _error_result(
            "format_type must be one of: iso, human, timestamp",
            format_type=format_type,
        )
    if not isinstance(check_if_within_working_hours, bool):
        return _error_result(
            "check_if_within_working_hours must be a boolean",
            check_if_within_working_hours=check_if_within_working_hours,
        )

    clean_format_type = format_type.strip()

    try:
        result = _API["get_current_time"](
            format_type=clean_format_type,
            check_if_within_working_hours=check_if_within_working_hours,
        )
        if hasattr(result, "__await__"):
            result = await result
        return {
            "status": "success",
            "value": result,
            "format_type": clean_format_type,
            "check_if_within_working_hours": check_if_within_working_hours,
            "fallback": _API["get_current_time"].__name__.endswith("fallback"),
        }
    except Exception as exc:
        return _error_result(
            str(exc),
            format_type=clean_format_type,
            check_if_within_working_hours=check_if_within_working_hours,
        )


def register_native_lizardpersons_function_tools(manager: Any) -> None:
    """Register native lizardpersons-function-tools category tools in unified manager."""
    manager.register_tool(
        category="lizardpersons_function_tools",
        name="get_current_time",
        func=get_current_time,
        description="Get current time from lizardperson function-tools compatibility layer.",
        input_schema={
            "type": "object",
            "properties": {
                "format_type": {
                    "type": "string",
                    "enum": ["iso", "human", "timestamp"],
                    "default": "iso",
                },
                "check_if_within_working_hours": {"type": "boolean", "default": False},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "lizardpersons-function-tools"],
    )
