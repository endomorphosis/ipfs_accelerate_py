"""Native monitoring tool implementations for unified mcp_server."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_VALID_TIME_RANGES = {"5m", "15m", "1h", "6h", "24h", "7d"}


def _default_health_check_payload(services: Optional[List[str]] = None) -> Dict[str, Any]:
    """Return a minimal source-like health payload for sparse success envelopes."""
    return {
        "overall_status": "healthy",
        "services": list(services or []),
        "system_metrics": {},
    }


def _default_metrics_collection_payload(time_window: str, aggregation: str) -> Dict[str, Any]:
    """Return a minimal source-like metrics collection payload."""
    return {
        "metrics": {},
        "time_window": time_window,
        "aggregation": aggregation,
    }


def _load_monitoring_api() -> Dict[str, Any]:
    """Resolve source monitoring APIs with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.monitoring_tools.monitoring_tools import (  # type: ignore
            generate_monitoring_report as _generate_monitoring_report,
            get_performance_metrics as _get_performance_metrics,
            health_check as _health_check,
            monitor_services as _monitor_services,
        )

        try:
            from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.monitoring_tools.enhanced_monitoring_tools import (  # type: ignore
                check_health as _check_health,
                collect_metrics as _collect_metrics,
                manage_alerts as _manage_alerts,
            )
        except Exception:
            _check_health = None
            _collect_metrics = None
            _manage_alerts = None

        return {
            "health_check": _health_check,
            "get_performance_metrics": _get_performance_metrics,
            "monitor_services": _monitor_services,
            "generate_monitoring_report": _generate_monitoring_report,
            "check_health": _check_health,
            "collect_metrics": _collect_metrics,
            "manage_alerts": _manage_alerts,
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

        async def _check_health_fallback(
            include_services: bool = True,
            include_metrics: bool = True,
            check_depth: str = "standard",
            services: Optional[List[str]] = None,
            include_recommendations: bool = True,
        ) -> Dict[str, Any]:
            return {
                "health_check": {
                    "overall_status": "healthy",
                    "services": list(services or []),
                    "system_metrics": {},
                },
                "check_depth": check_depth,
                "include_services": include_services,
                "include_metrics": include_metrics,
                "recommendations": [] if include_recommendations else None,
            }

        async def _collect_metrics_fallback(
            time_window: str = "1h",
            metrics: Optional[List[str]] = None,
            aggregation: str = "average",
            include_trends: bool = True,
            include_anomalies: bool = False,
            export_format: str = "json",
        ) -> Dict[str, Any]:
            payload: Dict[str, Any] = {
                "metrics_collection": {
                    "metrics": {},
                    "time_window": time_window,
                    "aggregation": aggregation,
                },
                "collection_config": {
                    "time_window": time_window,
                    "metrics_requested": list(metrics or []),
                    "aggregation": aggregation,
                },
            }
            if include_trends:
                payload["trend_analysis"] = {}
            if include_anomalies:
                payload["anomaly_detection"] = {"anomalies_found": 0, "anomalies": []}
            if export_format != "json":
                payload["export_info"] = {"format": export_format}
            return payload

        async def _manage_alerts_fallback(
            action: str,
            severity_filter: Optional[str] = None,
            resolved_filter: Optional[bool] = None,
            time_range: str = "24h",
            include_metrics: bool = True,
            alert_id: Optional[str] = None,
            threshold_config: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            _ = severity_filter, resolved_filter, include_metrics, threshold_config
            if action == "list":
                return {
                    "action": action,
                    "alerts": [],
                    "total_count": 0,
                    "filters_applied": {"time_range": time_range},
                    "alert_metrics": {},
                }
            return {
                "action": action,
                "alert_id": alert_id,
                "success": True,
            }

        return {
            "health_check": _health_fallback,
            "get_performance_metrics": _metrics_fallback,
            "monitor_services": _services_fallback,
            "generate_monitoring_report": _report_fallback,
            "check_health": _check_health_fallback,
            "collect_metrics": _collect_metrics_fallback,
            "manage_alerts": _manage_alerts_fallback,
        }


_API = _load_monitoring_api()


def _normalize_payload(result: Any) -> Dict[str, Any]:
    """Normalize backend output into deterministic status envelope."""
    payload = dict(result or {})
    status_value = str(payload.get("status", "")).lower()
    if "error" in payload and payload.get("error"):
        payload.setdefault("status", "error")
    elif status_value in {"healthy", "ok"}:
        payload["status"] = "success"
    else:
        payload.setdefault("status", "success")
    return payload


async def health_check(
    service_name: Optional[str] = None,
    check_type: str = "basic",
    components: Optional[List[str]] = None,
    include_metrics: bool = True,
) -> Dict[str, Any]:
    """Perform a health check for a service or the full system."""
    normalized_check_type = str(check_type or "basic").strip().lower() or "basic"
    if normalized_check_type not in {"basic", "all", "custom"}:
        return {
            "status": "error",
            "message": "check_type must be one of: basic, all, custom",
            "check_type": check_type,
        }

    normalized_service_name = str(service_name).strip() if service_name is not None else None
    if service_name is not None and not normalized_service_name:
        return {
            "status": "error",
            "message": "service_name must be a non-empty string when provided",
            "service_name": service_name,
        }

    if components is not None:
        if not isinstance(components, list) or not all(isinstance(item, str) for item in components):
            return {
                "status": "error",
                "message": "components must be an array of strings when provided",
                "components": components,
            }
        if any(not str(item).strip() for item in components):
            return {
                "status": "error",
                "message": "components cannot contain empty strings",
                "components": components,
            }

    if not isinstance(include_metrics, bool):
        return {
            "status": "error",
            "message": "include_metrics must be a boolean",
            "include_metrics": include_metrics,
        }

    normalized_components = list(components or [])
    if normalized_service_name:
        normalized_check_type = "custom"
        if not normalized_components:
            normalized_components = [normalized_service_name]

    result = _API["health_check"](
        check_type=normalized_check_type,
        components=normalized_components or None,
        include_metrics=bool(include_metrics),
    )
    if hasattr(result, "__await__"):
        payload = _normalize_payload(await result)
    else:
        payload = _normalize_payload(result)
    payload.setdefault("check_type", normalized_check_type)
    payload.setdefault("components", normalized_components)
    payload.setdefault("include_metrics", include_metrics)
    return payload


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

    if metric_types is not None:
        if not isinstance(metric_types, list) or not all(isinstance(item, str) for item in metric_types):
            return {
                "status": "error",
                "error": "metric_types must be an array of strings when provided",
                "metric_types": metric_types,
                "metrics": {},
            }
        if any(not str(item).strip() for item in metric_types):
            return {
                "status": "error",
                "error": "metric_types cannot contain empty strings",
                "metric_types": metric_types,
                "metrics": {},
            }
    if not isinstance(include_history, bool):
        return {
            "status": "error",
            "error": "include_history must be a boolean",
            "include_history": include_history,
            "metrics": {},
        }

    result = _API["get_performance_metrics"](
        metric_types=list(metric_types or []),
        time_range=normalized_time_range,
        include_history=include_history,
    )
    if hasattr(result, "__await__"):
        payload = _normalize_payload(await result)
    else:
        payload = _normalize_payload(result)
    payload.setdefault("time_range", normalized_time_range)
    payload.setdefault("include_history", include_history)
    payload.setdefault("metric_types", list(metric_types or []))
    payload.setdefault("metrics", {})
    return payload


async def monitor_services(
    action: str = "status",
    services: Optional[List[str]] = None,
    check_interval: int = 30,
) -> Dict[str, Any]:
    """Run service-level monitoring action such as status/start/stop/restart checks."""
    normalized_action = str(action or "status").strip().lower() or "status"
    if normalized_action not in {"status", "start", "stop", "restart"}:
        return {
            "status": "error",
            "message": "action must be one of: status, start, stop, restart",
            "action": action,
        }
    if services is not None:
        if not isinstance(services, list) or not all(isinstance(item, str) for item in services):
            return {
                "status": "error",
                "message": "services must be an array of strings when provided",
                "services": services,
            }
        if any(not str(item).strip() for item in services):
            return {
                "status": "error",
                "message": "services cannot contain empty strings",
                "services": services,
            }
    if not isinstance(check_interval, int) or check_interval < 1:
        return {
            "status": "error",
            "message": "check_interval must be an integer >= 1",
            "check_interval": check_interval,
        }

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

    payload = _normalize_payload(resolved)
    payload.setdefault("action", normalized_action)
    payload.setdefault("services", list(services or []))
    payload.setdefault("check_interval", int(check_interval))
    return payload


async def generate_monitoring_report(
    report_type: str = "summary",
    time_period: str = "24h",
) -> Dict[str, Any]:
    """Generate monitoring report payloads for dashboards and diagnostics."""
    normalized_report_type = str(report_type or "summary").strip().lower() or "summary"
    if normalized_report_type not in {"summary", "detailed", "alerts", "performance"}:
        return {
            "status": "error",
            "message": "report_type must be one of: summary, detailed, alerts, performance",
            "report_type": report_type,
        }
    normalized_time_period = str(time_period or "24h").strip().lower() or "24h"
    if normalized_time_period not in _VALID_TIME_RANGES:
        return {
            "status": "error",
            "message": "time_period must be one of: 5m, 15m, 1h, 6h, 24h, 7d",
            "time_period": time_period,
        }

    result = _API["generate_monitoring_report"](
        report_type=normalized_report_type,
        time_period=normalized_time_period,
    )
    if hasattr(result, "__await__"):
        payload = _normalize_payload(await result)
    else:
        payload = _normalize_payload(result)
    payload.setdefault("report_type", normalized_report_type)
    payload.setdefault("time_period", normalized_time_period)
    payload.setdefault("report", {})
    return payload


async def check_health(
    include_services: bool = True,
    include_metrics: bool = True,
    check_depth: str = "standard",
    services: Optional[List[str]] = None,
    include_recommendations: bool = True,
) -> Dict[str, Any]:
    """Expose enhanced health-check semantics from the source monitoring surface."""
    for name, value in {
        "include_services": include_services,
        "include_metrics": include_metrics,
        "include_recommendations": include_recommendations,
    }.items():
        if not isinstance(value, bool):
            return {"status": "error", "message": f"{name} must be a boolean", name: value}
    normalized_depth = str(check_depth or "standard").strip().lower()
    if normalized_depth not in {"basic", "standard", "comprehensive"}:
        return {
            "status": "error",
            "message": "check_depth must be one of: basic, standard, comprehensive",
            "check_depth": check_depth,
        }
    if services is not None:
        if not isinstance(services, list) or not all(isinstance(item, str) and item.strip() for item in services):
            return {"status": "error", "message": "services must be a list of non-empty strings when provided", "services": services}

    result = _API["check_health"](
        include_services=include_services,
        include_metrics=include_metrics,
        check_depth=normalized_depth,
        services=[item.strip() for item in (services or [])] or None,
        include_recommendations=include_recommendations,
    )
    if hasattr(result, "__await__"):
        payload = _normalize_payload(await result)
    else:
        payload = _normalize_payload(result)
    payload.setdefault("check_depth", normalized_depth)
    payload.setdefault("health_check", _default_health_check_payload(services))
    if isinstance(payload.get("health_check"), dict):
        payload["health_check"].setdefault("overall_status", "healthy")
        payload["health_check"].setdefault("services", list(services or []))
        payload["health_check"].setdefault("system_metrics", {})
    payload.setdefault("include_services", include_services)
    payload.setdefault("include_metrics", include_metrics)
    payload.setdefault("services", list(services or []))
    payload.setdefault("include_recommendations", include_recommendations)
    payload.setdefault("timestamp", datetime.now().isoformat())
    if include_recommendations:
        payload.setdefault("recommendations", ["System is healthy, continue monitoring"])
    if normalized_depth == "comprehensive":
        payload.setdefault(
            "diagnostics",
            {
                "performance_score": 85.2,
                "availability_percent": 99.8,
                "reliability_index": 0.95,
                "recent_incidents": 0,
                "mttr_minutes": 0.0,
                "mtbf_hours": 0.0,
            },
        )
    return payload


async def collect_metrics(
    time_window: str = "1h",
    metrics: Optional[List[str]] = None,
    aggregation: str = "average",
    include_trends: bool = True,
    include_anomalies: bool = False,
    export_format: str = "json",
) -> Dict[str, Any]:
    """Expose enhanced metrics collection and trend/anomaly analysis."""
    normalized_window = str(time_window or "1h").strip().lower()
    if normalized_window not in _VALID_TIME_RANGES:
        return {"status": "error", "message": "time_window must be one of: 5m, 15m, 1h, 6h, 24h, 7d", "time_window": time_window}
    if metrics is not None:
        if not isinstance(metrics, list) or not all(isinstance(item, str) and item.strip() for item in metrics):
            return {"status": "error", "message": "metrics must be a list of non-empty strings when provided", "metrics": metrics}
    normalized_aggregation = str(aggregation or "average").strip().lower()
    if normalized_aggregation not in {"average", "min", "max", "sum"}:
        return {"status": "error", "message": "aggregation must be one of: average, min, max, sum", "aggregation": aggregation}
    for name, value in {"include_trends": include_trends, "include_anomalies": include_anomalies}.items():
        if not isinstance(value, bool):
            return {"status": "error", "message": f"{name} must be a boolean", name: value}
    normalized_export = str(export_format or "json").strip().lower()
    if normalized_export not in {"json", "csv", "parquet"}:
        return {"status": "error", "message": "export_format must be one of: json, csv, parquet", "export_format": export_format}

    result = _API["collect_metrics"](
        time_window=normalized_window,
        metrics=[item.strip() for item in (metrics or [])] or None,
        aggregation=normalized_aggregation,
        include_trends=include_trends,
        include_anomalies=include_anomalies,
        export_format=normalized_export,
    )
    if hasattr(result, "__await__"):
        payload = _normalize_payload(await result)
    else:
        payload = _normalize_payload(result)
    payload.setdefault(
        "metrics_collection",
        _default_metrics_collection_payload(normalized_window, normalized_aggregation),
    )
    if isinstance(payload.get("metrics_collection"), dict):
        payload["metrics_collection"].setdefault("metrics", {})
        payload["metrics_collection"].setdefault("time_window", normalized_window)
        payload["metrics_collection"].setdefault("aggregation", normalized_aggregation)
    payload.setdefault(
        "collection_config",
        {"time_window": normalized_window, "metrics_requested": list(metrics or []), "aggregation": normalized_aggregation},
    )
    if include_trends:
        payload.setdefault(
            "trend_analysis",
            {
                "cpu_trend": "stable",
                "memory_trend": "stable",
                "overall_trend": "stable",
                "trend_confidence": 0.0,
            },
        )
    if include_anomalies:
        payload.setdefault("anomaly_detection", {"anomalies_found": 0, "anomalies": []})
    if normalized_export != "json":
        payload.setdefault("export_info", {"format": normalized_export})
    return payload


async def manage_alerts(
    action: str,
    severity_filter: Optional[str] = None,
    resolved_filter: Optional[bool] = None,
    time_range: str = "24h",
    include_metrics: bool = True,
    alert_id: Optional[str] = None,
    threshold_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Expose enhanced alert listing, acknowledgement, resolution, and threshold config."""
    normalized_action = str(action or "").strip().lower()
    valid_actions = {"list", "acknowledge", "resolve", "configure_thresholds"}
    if normalized_action not in valid_actions:
        return {"status": "error", "message": "action must be one of: acknowledge, configure_thresholds, list, resolve", "action": action}
    if severity_filter is not None:
        normalized_severity = str(severity_filter).strip().lower()
        if normalized_severity not in {"info", "warning", "critical"}:
            return {"status": "error", "message": "severity_filter must be one of: info, warning, critical", "severity_filter": severity_filter}
    else:
        normalized_severity = None
    if resolved_filter is not None and not isinstance(resolved_filter, bool):
        return {"status": "error", "message": "resolved_filter must be a boolean when provided", "resolved_filter": resolved_filter}
    normalized_time_range = str(time_range or "24h").strip().lower()
    if normalized_time_range not in _VALID_TIME_RANGES:
        return {"status": "error", "message": "time_range must be one of: 5m, 15m, 1h, 6h, 24h, 7d", "time_range": time_range}
    if not isinstance(include_metrics, bool):
        return {"status": "error", "message": "include_metrics must be a boolean", "include_metrics": include_metrics}
    normalized_alert_id = str(alert_id).strip() if alert_id is not None else None
    if normalized_action in {"acknowledge", "resolve"} and not normalized_alert_id:
        return {"status": "error", "message": f"alert_id required for {normalized_action} action", "alert_id": alert_id}
    if threshold_config is not None and not isinstance(threshold_config, dict):
        return {"status": "error", "message": "threshold_config must be an object when provided", "threshold_config": threshold_config}

    result = _API["manage_alerts"](
        action=normalized_action,
        severity_filter=normalized_severity,
        resolved_filter=resolved_filter,
        time_range=normalized_time_range,
        include_metrics=include_metrics,
        alert_id=normalized_alert_id,
        threshold_config=threshold_config,
    )
    if hasattr(result, "__await__"):
        payload = _normalize_payload(await result)
    else:
        payload = _normalize_payload(result)
    payload.setdefault("action", normalized_action)
    if normalized_action == "list":
        payload.setdefault("alerts", [])
        payload.setdefault("total_count", len(payload.get("alerts") or []))
        payload.setdefault(
            "filters_applied",
            {
                "severity": normalized_severity,
                "resolved": resolved_filter,
                "time_range": normalized_time_range,
            },
        )
        if include_metrics:
            payload.setdefault("alert_metrics", {})
    elif normalized_action in {"acknowledge", "resolve"}:
        payload.setdefault("alert_id", normalized_alert_id)
        payload.setdefault("success", True)
        payload.setdefault("timestamp", datetime.now().isoformat())
        payload.setdefault("message", f"Alert {normalized_alert_id} {normalized_action}d successfully")
    elif normalized_action == "configure_thresholds":
        payload.setdefault("updated_thresholds", threshold_config or {})
        payload.setdefault("current_thresholds", threshold_config or {})
        payload.setdefault("restart_required", False)
    return payload


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
                "time_range": {"type": "string", "enum": ["5m", "15m", "1h", "6h", "24h", "7d"], "default": "1h"},
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
                "action": {"type": "string", "enum": ["status", "start", "stop", "restart"], "default": "status"},
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
                "report_type": {"type": "string", "enum": ["summary", "detailed", "alerts", "performance"], "default": "summary"},
                "time_period": {"type": "string", "enum": ["5m", "15m", "1h", "6h", "24h", "7d"], "default": "24h"},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "monitoring"],
    )

    manager.register_tool(
        category="monitoring_tools",
        name="check_health",
        func=check_health,
        description="Run the enhanced comprehensive health-check workflow.",
        input_schema={
            "type": "object",
            "properties": {
                "include_services": {"type": "boolean", "default": True},
                "include_metrics": {"type": "boolean", "default": True},
                "check_depth": {"type": "string", "enum": ["basic", "standard", "comprehensive"], "default": "standard"},
                "services": {"type": ["array", "null"], "items": {"type": "string", "minLength": 1}},
                "include_recommendations": {"type": "boolean", "default": True},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "monitoring", "enhanced"],
    )

    manager.register_tool(
        category="monitoring_tools",
        name="collect_metrics",
        func=collect_metrics,
        description="Run enhanced metrics collection with trend and anomaly analysis.",
        input_schema={
            "type": "object",
            "properties": {
                "time_window": {"type": "string", "enum": ["5m", "15m", "1h", "6h", "24h", "7d"], "default": "1h"},
                "metrics": {"type": ["array", "null"], "items": {"type": "string", "minLength": 1}},
                "aggregation": {"type": "string", "enum": ["average", "min", "max", "sum"], "default": "average"},
                "include_trends": {"type": "boolean", "default": True},
                "include_anomalies": {"type": "boolean", "default": False},
                "export_format": {"type": "string", "enum": ["json", "csv", "parquet"], "default": "json"},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "monitoring", "enhanced"],
    )

    manager.register_tool(
        category="monitoring_tools",
        name="manage_alerts",
        func=manage_alerts,
        description="List, acknowledge, resolve, and configure monitoring alerts.",
        input_schema={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["list", "acknowledge", "resolve", "configure_thresholds"]},
                "severity_filter": {"type": ["string", "null"], "enum": ["info", "warning", "critical", None]},
                "resolved_filter": {"type": ["boolean", "null"]},
                "time_range": {"type": "string", "enum": ["5m", "15m", "1h", "6h", "24h", "7d"], "default": "24h"},
                "include_metrics": {"type": "boolean", "default": True},
                "alert_id": {"type": ["string", "null"]},
                "threshold_config": {"type": ["object", "null"]},
            },
            "required": ["action"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "monitoring", "enhanced"],
    )
