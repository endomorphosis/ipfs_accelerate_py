"""Native monitoring tool implementations for unified mcp_server."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


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

        async def _health_fallback(service_name: Optional[str] = None) -> Dict[str, Any]:
            return {
                "status": "healthy",
                "service": service_name or "all",
                "message": "Fallback health check used",
            }

        async def _metrics_fallback(time_range: str = "1h") -> Dict[str, Any]:
            return {
                "status": "success",
                "time_range": time_range,
                "metrics": {
                    "cpu": {"avg": 0.0, "max": 0.0},
                    "memory": {"avg": 0.0, "max": 0.0},
                },
            }

        async def _services_fallback(action: str = "status") -> Dict[str, Any]:
            return {
                "status": "success",
                "action": action,
                "services": [],
            }

        async def _report_fallback(report_type: str = "summary") -> Dict[str, Any]:
            return {
                "status": "success",
                "report_type": report_type,
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


async def health_check(service_name: Optional[str] = None) -> Dict[str, Any]:
    """Perform a health check for a service or the full system."""
    result = _API["health_check"](service_name=service_name)
    if hasattr(result, "__await__"):
        return await result
    return result


async def get_performance_metrics(time_range: str = "1h") -> Dict[str, Any]:
    """Return aggregated performance metrics for the given time range."""
    result = _API["get_performance_metrics"](time_range=time_range)
    if hasattr(result, "__await__"):
        return await result
    return result


async def monitor_services(action: str = "status") -> Dict[str, Any]:
    """Run service-level monitoring action such as status/start/stop/restart checks."""
    result = _API["monitor_services"](action=action)
    if hasattr(result, "__await__"):
        return await result
    return result


async def generate_monitoring_report(report_type: str = "summary") -> Dict[str, Any]:
    """Generate monitoring report payloads for dashboards and diagnostics."""
    result = _API["generate_monitoring_report"](report_type=report_type)
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
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "monitoring"],
    )
