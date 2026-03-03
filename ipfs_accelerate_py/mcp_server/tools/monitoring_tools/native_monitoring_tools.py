"""Native monitoring tool implementations for unified mcp_server."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_VALID_TIME_RANGES = {"5m", "15m", "1h", "6h", "24h", "7d"}


def _load_monitoring_api() -> Dict[str, Any]:
    """Resolve source monitoring APIs with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.monitoring_tools.monitoring_tools import (  # type: ignore
            generate_monitoring_report as _generate_monitoring_report,
            get_performance_metrics as _get_performance_metrics,
            health_check as _health_check,
            monitor_services as _monitor_services,
        )

        return {
            "health_check": _health_check,
            "get_performance_metrics": _get_performance_metrics,
            "monitor_services": _monitor_services,
            "generate_monitoring_report": _generate_monitoring_report,
        }
    except Exception:
        logger.warning("Source monitoring_tools import unavailable, using fallback monitoring functions")

        async def _health_fallback(
            check_type: str = "basic",
            components: Optional[List[str]] = None,
            include_metrics: bool = True,
        ) -> Dict[str, Any]:
            if check_type == "custom" and components:
                service_label = ",".join(str(c) for c in components)
            else:
                service_label = "all"
            return {
                "status": "healthy",
                "service": service_label,
                "message": "Fallback health check used",
                "check_type": check_type,
                "include_metrics": bool(include_metrics),
            }

        async def _metrics_fallback(
            metric_types: Optional[List[str]] = None,
            time_range: str = "1h",
            include_history: bool = True,
        ) -> Dict[str, Any]:
            return {
                "status": "success",
                "time_range": time_range,
                "metric_types": list(metric_types or []),
                "include_history": bool(include_history),
                "metrics": {
                    "cpu": {"avg": 0.0, "max": 0.0},
                    "memory": {"avg": 0.0, "max": 0.0},
                },
            }

        async def _services_fallback(
            services: Optional[List[str]] = None,
            check_interval: int = 30,
        ) -> Dict[str, Any]:
            return {
                "status": "success",
                "services": list(services or []),
                "check_interval": int(check_interval),
            }

        async def _report_fallback(
            report_type: str = "summary",
            time_period: str = "24h",
        ) -> Dict[str, Any]:
            return {
                "status": "success",
                "report_type": report_type,
                "time_period": time_period,
                "report": {
                    "summary": "Fallback monitoring report",
                },
            }

        return {
            "health_check": _health_fallback,
            "get_performance_metrics": _metrics_fallback,
            "monitor_services": _services_fallback,
            "generate_monitoring_report": _report_fallback,
        }


_API = _load_monitoring_api()


async def health_check(
    service_name: Optional[str] = None,
    check_type: str = "basic",
    components: Optional[List[str]] = None,
    include_metrics: bool = True,
) -> Dict[str, Any]:
    """Perform a health check for a service or the full system."""
    normalized_check_type = str(check_type or "basic").strip().lower() or "basic"
    normalized_components = list(components or [])
    if service_name:
        normalized_check_type = "custom"
        if not normalized_components:
            normalized_components = [str(service_name).strip()]

    result = _API["health_check"](
        check_type=normalized_check_type,
        components=normalized_components or None,
        include_metrics=bool(include_metrics),
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def get_performance_metrics(
    time_range: str = "1h",
    metric_types: Optional[List[str]] = None,
    include_history: bool = True,
) -> Dict[str, Any]:
    """Return aggregated performance metrics for the given time range."""
    normalized_time_range = str(time_range or "1h").strip().lower() or "1h"
    if normalized_time_range not in _VALID_TIME_RANGES:
        return {
            "status": "error",
            "error": (
                "time_range must be one of: " + ", ".join(sorted(_VALID_TIME_RANGES))
            ),
            "time_range": normalized_time_range,
            "metrics": {},
        }

    result = _API["get_performance_metrics"](
        metric_types=list(metric_types or []),
        time_range=normalized_time_range,
        include_history=bool(include_history),
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def monitor_services(
    action: str = "status",
    services: Optional[List[str]] = None,
    check_interval: int = 30,
) -> Dict[str, Any]:
    """Run service-level monitoring action such as status/start/stop/restart checks."""
    normalized_action = str(action or "status").strip().lower() or "status"
    if normalized_action != "status":
        # Source API only exposes status checks; preserve legacy action field deterministically.
        return {
            "status": "success",
            "action": normalized_action,
            "message": "action not supported by source monitoring API; status snapshot not executed",
            "services": list(services or []),
            "check_interval": int(check_interval),
        }

    result = _API["monitor_services"](
        services=list(services or []),
        check_interval=int(check_interval),
    )
    if hasattr(result, "__await__"):
        resolved = await result
    else:
        resolved = result

    if isinstance(resolved, dict):
        resolved.setdefault("action", normalized_action)
    return resolved


async def generate_monitoring_report(
    report_type: str = "summary",
    time_period: str = "24h",
) -> Dict[str, Any]:
    """Generate monitoring report payloads for dashboards and diagnostics."""
    result = _API["generate_monitoring_report"](
        report_type=str(report_type or "summary").strip().lower() or "summary",
        time_period=str(time_period or "24h").strip().lower() or "24h",
    )
    if hasattr(result, "__await__"):
        return await result
    return result


def register_native_monitoring_tools(manager: Any) -> None:
    """Register native monitoring tools in unified hierarchical manager."""
    manager.register_tool(
        category="monitoring_tools",
        name="health_check",
        func=health_check,
        description="Check health for all services or a specific service.",
        input_schema={
            "type": "object",
            "properties": {
                "service_name": {"type": ["string", "null"]},
                "check_type": {
                    "type": "string",
                    "enum": ["basic", "all", "custom"],
                    "default": "basic",
                },
                "components": {"type": ["array", "null"], "items": {"type": "string"}},
                "include_metrics": {"type": "boolean", "default": True},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "monitoring"],
    )

    manager.register_tool(
        category="monitoring_tools",
        name="get_performance_metrics",
        func=get_performance_metrics,
        description="Retrieve performance metrics for a selected time window.",
        input_schema={
            "type": "object",
            "properties": {
                "time_range": {"type": "string"},
                "metric_types": {"type": ["array", "null"], "items": {"type": "string"}},
                "include_history": {"type": "boolean", "default": True},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "monitoring"],
    )

    manager.register_tool(
        category="monitoring_tools",
        name="monitor_services",
        func=monitor_services,
        description="Manage and inspect monitored service state.",
        input_schema={
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "services": {"type": ["array", "null"], "items": {"type": "string"}},
                "check_interval": {"type": "integer", "minimum": 1, "default": 30},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "monitoring"],
    )

    manager.register_tool(
        category="monitoring_tools",
        name="generate_monitoring_report",
        func=generate_monitoring_report,
        description="Generate summary or detailed monitoring reports.",
        input_schema={
            "type": "object",
            "properties": {
                "report_type": {"type": "string"},
                "time_period": {"type": "string", "default": "24h"},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "monitoring"],
    )
