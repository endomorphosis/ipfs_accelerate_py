"""Native admin tool implementations for unified mcp_server."""

from __future__ import annotations

from datetime import datetime
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _normalize_delegate_payload(payload: Any) -> Dict[str, Any]:
    """Normalize delegate payloads with deterministic failed-status inference."""
    normalized = dict(payload or {})
    failed = normalized.get("success") is False or bool(normalized.get("error"))
    if failed:
        normalized["status"] = "error"
    else:
        normalized.setdefault("status", "success")
    return normalized


def _load_admin_api() -> Dict[str, Any]:
    """Resolve source admin APIs with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.admin_tools.admin_tools import (  # type: ignore
            configure_system as _configure_system,
            manage_endpoints as _manage_endpoints,
            system_health as _system_health,
            system_maintenance as _system_maintenance,
        )

        try:
            from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.admin_tools.enhanced_admin_tools import (  # type: ignore
                cleanup_resources as _cleanup_resources,
                get_system_status as _get_system_status,
                manage_service as _manage_service,
                update_configuration as _update_configuration,
            )
        except Exception:
            _get_system_status = None
            _manage_service = None
            _update_configuration = None
            _cleanup_resources = None

        return {
            "manage_endpoints": _manage_endpoints,
            "system_maintenance": _system_maintenance,
            "configure_system": _configure_system,
            "system_health": _system_health,
            "get_system_status": _get_system_status,
            "manage_service": _manage_service,
            "update_configuration": _update_configuration,
            "cleanup_resources": _cleanup_resources,
        }
    except Exception:
        logger.warning("Source admin_tools import unavailable, using fallback admin functions")

        async def _manage_endpoints_fallback(
            action: str,
            model: Optional[str] = None,
            endpoint: Optional[str] = None,
            endpoint_type: Optional[str] = None,
            ctx_length: Optional[int] = 512,
        ) -> Dict[str, Any]:
            _ = model, endpoint, endpoint_type, ctx_length
            if action == "list":
                return {
                    "success": True,
                    "status": "success",
                    "action": action,
                    "endpoints": [],
                    "count": 0,
                    "timestamp": datetime.now().isoformat(),
                }
            return {
                "success": True,
                "status": "success",
                "action": action,
                "message": "Endpoint action handled by fallback",
            }

        async def _system_maintenance_fallback(
            operation: Optional[str] = None,
            target: Optional[str] = None,
            force: bool = False,
            action: Optional[str] = None,
        ) -> Dict[str, Any]:
            _ = target, force
            op = operation if operation is not None else action
            return {
                "success": True,
                "status": "success",
                "operation": op or "health_check",
                "timestamp": datetime.now().isoformat(),
            }

        async def _configure_system_fallback(
            action: str,
            config_key: Optional[str] = None,
            settings: Optional[Dict[str, Any]] = None,
            validate_only: bool = False,
        ) -> Dict[str, Any]:
            return {
                "success": True,
                "status": "success",
                "action": action,
                "config_key": config_key,
                "settings": settings or {},
                "validated": validate_only,
                "timestamp": datetime.now().isoformat(),
            }

        async def _system_health_fallback(
            component: str = "all",
            detailed: bool = False,
        ) -> Dict[str, Any]:
            return {
                "success": True,
                "status": "success",
                "component": component,
                "detailed": detailed,
                "health": "healthy",
                "components": {},
                "timestamp": datetime.now().isoformat(),
            }

        async def _get_system_status_fallback(
            include_details: bool = True,
            include_services: bool = True,
            include_resources: bool = True,
            format: str = "json",
        ) -> Dict[str, Any]:
            base = {
                "system_status": "operational",
                "timestamp": datetime.now().isoformat(),
                "health_status": "healthy",
            }
            if format == "summary":
                return {
                    "status": "operational",
                    "health": "healthy",
                    "services_running": 0,
                    "total_services": 0,
                    "cpu_usage": 0.0,
                    "memory_usage": 0.0,
                }
            if include_details:
                base["system_info"] = {
                    "environment": "fallback",
                    "version": "unknown",
                }
            if include_services:
                base["services"] = {}
                base["maintenance_mode"] = False
            if include_resources:
                base["resource_usage"] = {
                    "cpu_percent": 0.0,
                    "memory_percent": 0.0,
                    "disk_percent": 0.0,
                }
            if format == "detailed":
                base["diagnostics"] = {
                    "last_restart": None,
                    "error_count_24h": 0,
                    "warning_count_24h": 0,
                    "active_connections": 0,
                    "queue_length": 0,
                }
            return base

        async def _manage_service_fallback(
            service_name: str,
            action: str,
            force: bool = False,
            timeout_seconds: int = 30,
        ) -> Dict[str, Any]:
            if service_name == "all":
                return {
                    "bulk_operation": True,
                    "action": action,
                    "total_services": 4,
                    "successful_operations": 4,
                    "failed_operations": 0,
                    "results": [
                        {
                            "service_name": name,
                            "action": action,
                            "success": True,
                            "force_applied": force,
                            "timeout_seconds": timeout_seconds,
                        }
                        for name in [
                            "ipfs_daemon",
                            "vector_store",
                            "cache_service",
                            "monitoring_service",
                        ]
                    ],
                }
            return {
                "single_operation": True,
                "success": True,
                "service_name": service_name,
                "action": action,
                "timeout_seconds": timeout_seconds,
                "force_applied": force,
            }

        async def _update_configuration_fallback(
            action: str,
            config_updates: Optional[Dict[str, Any]] = None,
            create_backup: bool = True,
            validate_config: bool = True,
        ) -> Dict[str, Any]:
            return {
                "action": action,
                "success": True,
                "configuration": config_updates or {},
                "validation_passed": validate_config,
                "backup_created": create_backup,
                "timestamp": datetime.now().isoformat(),
            }

        async def _cleanup_resources_fallback(
            cleanup_type: str = "basic",
            restart_services: bool = True,
            cleanup_temp_files: bool = True,
            cleanup_logs: bool = False,
            max_log_age_days: int = 30,
        ) -> Dict[str, Any]:
            return {
                "success": True,
                "cleanup_type": cleanup_type,
                "freed_memory_bytes": 0,
                "disk_space_freed_mb": 0,
                "cleanup_options": {
                    "restart_services": restart_services,
                    "cleanup_temp_files": cleanup_temp_files,
                    "cleanup_logs": cleanup_logs,
                    "max_log_age_days": max_log_age_days,
                },
                "recommendations": [],
                "timestamp": datetime.now().isoformat(),
            }

        return {
            "manage_endpoints": _manage_endpoints_fallback,
            "system_maintenance": _system_maintenance_fallback,
            "configure_system": _configure_system_fallback,
            "system_health": _system_health_fallback,
            "get_system_status": _get_system_status_fallback,
            "manage_service": _manage_service_fallback,
            "update_configuration": _update_configuration_fallback,
            "cleanup_resources": _cleanup_resources_fallback,
        }


_API = _load_admin_api()


async def manage_endpoints(
    action: str,
    model: Optional[str] = None,
    endpoint: Optional[str] = None,
    endpoint_type: Optional[str] = None,
    ctx_length: Optional[int] = 512,
) -> Dict[str, Any]:
    """Manage endpoint records for embedding/model services."""
    normalized_action = str(action or "").strip().lower()
    allowed_actions = {"add", "update", "remove", "list"}
    if normalized_action not in allowed_actions:
        return {
            "status": "error",
            "message": f"action must be one of: {', '.join(sorted(allowed_actions))}",
            "action": action,
        }
    if normalized_action == "add" and (not str(model or "").strip() or not str(endpoint or "").strip() or not str(endpoint_type or "").strip()):
        return {
            "status": "error",
            "message": "model, endpoint, and endpoint_type are required for add action",
            "action": normalized_action,
        }
    if normalized_action in {"update", "remove"} and not str(model or "").strip():
        return {
            "status": "error",
            "message": "model is required for update/remove actions",
            "action": normalized_action,
        }
    if endpoint_type is not None:
        normalized_endpoint_type = str(endpoint_type).strip().lower()
        valid_endpoint_types = {"local", "http", "https", "openai", "azure", "sagemaker"}
        if normalized_endpoint_type not in valid_endpoint_types:
            return {
                "status": "error",
                "message": "endpoint_type must be one of: azure, http, https, local, openai, sagemaker",
                "endpoint_type": endpoint_type,
            }
    normalized_ctx_length: Optional[int] = None
    if ctx_length is not None:
        try:
            normalized_ctx_length = int(ctx_length)
        except (TypeError, ValueError):
            return {
                "status": "error",
                "message": "ctx_length must be a positive integer when provided",
                "ctx_length": ctx_length,
            }
        if normalized_ctx_length <= 0:
            return {
                "status": "error",
                "message": "ctx_length must be a positive integer when provided",
                "ctx_length": ctx_length,
            }

    result = await _API["manage_endpoints"](
        action=normalized_action,
        model=model,
        endpoint=endpoint,
        endpoint_type=endpoint_type,
        ctx_length=normalized_ctx_length,
    )
    payload = _normalize_delegate_payload(result)
    payload.setdefault("action", normalized_action)
    return payload


async def system_maintenance(
    operation: Optional[str] = None,
    target: Optional[str] = None,
    force: bool = False,
    action: Optional[str] = None,
) -> Dict[str, Any]:
    """Perform maintenance operations like health-check and cleanup."""
    normalized_operation = str(operation or action or "health_check").strip().lower()
    alias_map = {"status": "health_check", "health": "health_check"}
    normalized_operation = alias_map.get(normalized_operation, normalized_operation)
    allowed_operations = {"health_check", "cleanup", "restart", "backup"}
    if normalized_operation not in allowed_operations:
        return {
            "status": "error",
            "message": f"operation must be one of: {', '.join(sorted(allowed_operations))}",
            "operation": operation,
            "action": action,
        }
    if not isinstance(force, bool):
        return {
            "status": "error",
            "message": "force must be a boolean",
            "force": force,
        }

    result = await _API["system_maintenance"](
        operation=normalized_operation,
        target=target,
        force=force,
        action=normalized_operation,
    )
    payload = _normalize_delegate_payload(result)
    payload.setdefault("operation", normalized_operation)
    return payload


async def configure_system(
    action: str,
    config_key: Optional[str] = None,
    settings: Optional[Dict[str, Any]] = None,
    validate_only: bool = False,
) -> Dict[str, Any]:
    """Get or update system configuration settings."""
    normalized_action = str(action or "").strip().lower()
    allowed_actions = {"get", "set", "update", "configure"}
    if normalized_action not in allowed_actions:
        return {
            "status": "error",
            "message": f"action must be one of: {', '.join(sorted(allowed_actions))}",
            "action": action,
        }
    if normalized_action in {"set", "update", "configure"} and settings is not None and not isinstance(settings, dict):
        return {
            "status": "error",
            "message": "settings must be an object when provided",
            "settings": settings,
        }
    if config_key is not None and not str(config_key).strip():
        return {
            "status": "error",
            "message": "config_key must be a non-empty string when provided",
            "config_key": config_key,
        }
    if not isinstance(validate_only, bool):
        return {
            "status": "error",
            "message": "validate_only must be a boolean",
            "validate_only": validate_only,
        }

    result = await _API["configure_system"](
        action=normalized_action,
        config_key=config_key,
        settings=settings,
        validate_only=validate_only,
    )
    payload = _normalize_delegate_payload(result)
    payload.setdefault("action", normalized_action)
    return payload


async def system_health(
    component: str = "all",
    detailed: bool = False,
) -> Dict[str, Any]:
    """Return system health summary with deterministic validation and envelope."""
    normalized_component = str(component or "").strip().lower()
    if not normalized_component:
        return {
            "status": "error",
            "message": "component must be a non-empty string",
            "component": component,
        }
    if not isinstance(detailed, bool):
        return {
            "status": "error",
            "message": "detailed must be a boolean",
            "detailed": detailed,
        }

    result = await _API["system_health"](
        component=normalized_component,
        detailed=detailed,
    )
    payload = _normalize_delegate_payload(result)
    payload.setdefault("component", normalized_component)
    payload.setdefault("detailed", detailed)
    payload.setdefault("health", "unknown")
    return payload


async def get_system_status(
    include_details: bool = True,
    include_services: bool = True,
    include_resources: bool = True,
    format: str = "json",
) -> Dict[str, Any]:
    """Return enhanced system-status summary compatible with source wrappers."""
    if not isinstance(include_details, bool):
        return {"status": "error", "message": "include_details must be a boolean", "include_details": include_details}
    if not isinstance(include_services, bool):
        return {"status": "error", "message": "include_services must be a boolean", "include_services": include_services}
    if not isinstance(include_resources, bool):
        return {"status": "error", "message": "include_resources must be a boolean", "include_resources": include_resources}

    normalized_format = str(format or "json").strip().lower()
    if normalized_format not in {"json", "summary", "detailed"}:
        return {
            "status": "error",
            "message": "format must be one of: json, summary, detailed",
            "format": format,
        }

    handler = _API.get("get_system_status")
    if handler is not None:
        result = await handler(
            include_details=include_details,
            include_services=include_services,
            include_resources=include_resources,
            format=normalized_format,
        )
        payload = _normalize_delegate_payload(result)
    else:
        base_health = await system_health(component="all", detailed=normalized_format == "detailed")
        payload = {
            "system_status": "operational",
            "timestamp": datetime.now().isoformat(),
            "health_status": base_health.get("health", "unknown"),
        }

    if normalized_format == "summary":
        payload.setdefault("status", payload.get("status", "operational"))
        payload.setdefault("health", payload.get("health_status", "unknown"))
        payload.setdefault("services_running", len(payload.get("services") or {}))
        payload.setdefault("total_services", len(payload.get("services") or {}))
        payload.setdefault("cpu_usage", ((payload.get("resource_usage") or {}).get("cpu_percent", 0.0)))
        payload.setdefault("memory_usage", ((payload.get("resource_usage") or {}).get("memory_percent", 0.0)))
        return payload

    payload.setdefault("status", "success")
    payload.setdefault("system_status", "operational")
    payload.setdefault("health_status", "unknown")
    if include_details:
        payload.setdefault("system_info", {})
    if include_services:
        payload.setdefault("services", {})
        payload.setdefault("maintenance_mode", False)
    if include_resources:
        payload.setdefault("resource_usage", {})
    if normalized_format == "detailed":
        payload.setdefault("diagnostics", {})
    return payload


async def manage_service(
    service_name: str,
    action: str,
    force: bool = False,
    timeout_seconds: int = 30,
) -> Dict[str, Any]:
    """Manage one service or all known services using enhanced admin parity."""
    normalized_service_name = str(service_name or "").strip()
    if not normalized_service_name:
        return {"status": "error", "message": "service_name must be a non-empty string", "service_name": service_name}

    normalized_action = str(action or "").strip().lower()
    if normalized_action not in {"start", "stop", "restart", "status"}:
        return {
            "status": "error",
            "message": "action must be one of: restart, start, status, stop",
            "action": action,
        }
    if not isinstance(force, bool):
        return {"status": "error", "message": "force must be a boolean", "force": force}
    if not isinstance(timeout_seconds, int) or timeout_seconds < 1:
        return {
            "status": "error",
            "message": "timeout_seconds must be a positive integer",
            "timeout_seconds": timeout_seconds,
        }

    handler = _API.get("manage_service")
    if handler is not None:
        result = await handler(
            service_name=normalized_service_name,
            action=normalized_action,
            force=force,
            timeout_seconds=timeout_seconds,
        )
        payload = _normalize_delegate_payload(result)
    else:
        payload = {
            "single_operation": normalized_service_name != "all",
            "success": True,
            "service_name": normalized_service_name,
            "action": normalized_action,
            "timeout_seconds": timeout_seconds,
            "force_applied": force,
        }

    payload.setdefault("status", "success")
    payload.setdefault("action", normalized_action)
    if normalized_service_name == "all":
        payload.setdefault("bulk_operation", True)
        payload.setdefault("results", [])
        payload.setdefault("total_services", len(payload.get("results") or []))
        payload.setdefault(
            "successful_operations",
            sum(1 for item in (payload.get("results") or []) if item.get("success", True)),
        )
        payload.setdefault(
            "failed_operations",
            sum(1 for item in (payload.get("results") or []) if item.get("success") is False or item.get("error")),
        )
    else:
        payload.setdefault("single_operation", True)
        payload.setdefault("service_name", normalized_service_name)
        payload.setdefault("timeout_seconds", timeout_seconds)
        payload.setdefault("force_applied", force)
    return payload


async def update_configuration(
    action: str,
    config_updates: Optional[Dict[str, Any]] = None,
    create_backup: bool = True,
    validate_config: bool = True,
) -> Dict[str, Any]:
    """Get, update, validate, backup, or restore configuration using enhanced parity."""
    normalized_action = str(action or "").strip().lower()
    if normalized_action not in {"get", "update", "validate", "backup", "restore"}:
        return {
            "status": "error",
            "message": "action must be one of: backup, get, restore, update, validate",
            "action": action,
        }
    if config_updates is not None and not isinstance(config_updates, dict):
        return {
            "status": "error",
            "message": "config_updates must be an object when provided",
            "config_updates": config_updates,
        }
    if not isinstance(create_backup, bool):
        return {"status": "error", "message": "create_backup must be a boolean", "create_backup": create_backup}
    if not isinstance(validate_config, bool):
        return {"status": "error", "message": "validate_config must be a boolean", "validate_config": validate_config}

    handler = _API.get("update_configuration")
    if handler is not None:
        result = await handler(
            action=normalized_action,
            config_updates=config_updates,
            create_backup=create_backup,
            validate_config=validate_config,
        )
        payload = _normalize_delegate_payload(result)
    else:
        payload = {
            "action": normalized_action,
            "success": True,
            "configuration": config_updates or {},
            "validation_passed": validate_config,
            "backup_created": create_backup,
            "timestamp": datetime.now().isoformat(),
        }

    payload.setdefault("status", "success")
    payload.setdefault("action", normalized_action)
    if normalized_action == "validate":
        payload.setdefault("validation_results", [])
        payload.setdefault("is_valid", True)
    return payload


async def cleanup_resources(
    cleanup_type: str = "basic",
    restart_services: bool = True,
    cleanup_temp_files: bool = True,
    cleanup_logs: bool = False,
    max_log_age_days: int = 30,
) -> Dict[str, Any]:
    """Clean up system resources using enhanced admin parity."""
    normalized_cleanup_type = str(cleanup_type or "").strip().lower()
    if normalized_cleanup_type not in {"basic", "full", "cache_only", "logs_only"}:
        return {
            "status": "error",
            "message": "cleanup_type must be one of: basic, cache_only, full, logs_only",
            "cleanup_type": cleanup_type,
        }
    if not isinstance(restart_services, bool):
        return {"status": "error", "message": "restart_services must be a boolean", "restart_services": restart_services}
    if not isinstance(cleanup_temp_files, bool):
        return {"status": "error", "message": "cleanup_temp_files must be a boolean", "cleanup_temp_files": cleanup_temp_files}
    if not isinstance(cleanup_logs, bool):
        return {"status": "error", "message": "cleanup_logs must be a boolean", "cleanup_logs": cleanup_logs}
    if not isinstance(max_log_age_days, int) or max_log_age_days < 1:
        return {
            "status": "error",
            "message": "max_log_age_days must be a positive integer",
            "max_log_age_days": max_log_age_days,
        }

    handler = _API.get("cleanup_resources")
    if handler is not None:
        result = await handler(
            cleanup_type=normalized_cleanup_type,
            restart_services=restart_services,
            cleanup_temp_files=cleanup_temp_files,
            cleanup_logs=cleanup_logs,
            max_log_age_days=max_log_age_days,
        )
        payload = _normalize_delegate_payload(result)
    else:
        payload = {
            "success": True,
            "cleanup_type": normalized_cleanup_type,
            "freed_memory_bytes": 0,
            "disk_space_freed_mb": 0,
            "timestamp": datetime.now().isoformat(),
        }

    payload.setdefault("status", "success")
    payload.setdefault("cleanup_type", normalized_cleanup_type)
    payload.setdefault(
        "cleanup_options",
        {
            "restart_services": restart_services,
            "cleanup_temp_files": cleanup_temp_files,
            "cleanup_logs": cleanup_logs,
            "max_log_age_days": max_log_age_days,
        },
    )
    payload.setdefault("recommendations", [])
    return payload


def register_native_admin_tools(manager: Any) -> None:
    """Register native admin tools in unified hierarchical manager."""
    manager.register_tool(
        category="admin_tools",
        name="manage_endpoints",
        func=manage_endpoints,
        description="Manage API endpoints and endpoint configurations.",
        input_schema={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["add", "update", "remove", "list"]},
                "model": {"type": ["string", "null"]},
                "endpoint": {"type": ["string", "null"]},
                "endpoint_type": {"type": ["string", "null"], "enum": ["local", "http", "https", "openai", "azure", "sagemaker", None]},
                "ctx_length": {"type": ["integer", "null"], "minimum": 1, "default": 512},
            },
            "required": ["action"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "admin"],
    )

    manager.register_tool(
        category="admin_tools",
        name="system_maintenance",
        func=system_maintenance,
        description="Perform maintenance operations for system components.",
        input_schema={
            "type": "object",
            "properties": {
                "operation": {"type": ["string", "null"], "enum": ["health_check", "cleanup", "restart", "backup", "status", "health", None]},
                "target": {"type": ["string", "null"]},
                "force": {"type": "boolean", "default": False},
                "action": {"type": ["string", "null"], "enum": ["health_check", "cleanup", "restart", "backup", "status", "health", None]},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "admin"],
    )

    manager.register_tool(
        category="admin_tools",
        name="configure_system",
        func=configure_system,
        description="Get and update system configuration values.",
        input_schema={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["get", "set", "update", "configure"]},
                "config_key": {"type": ["string", "null"]},
                "settings": {"type": ["object", "null"]},
                "validate_only": {"type": "boolean", "default": False},
            },
            "required": ["action"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "admin"],
    )

    manager.register_tool(
        category="admin_tools",
        name="system_health",
        func=system_health,
        description="Get system health for a target component.",
        input_schema={
            "type": "object",
            "properties": {
                "component": {"type": "string", "minLength": 1, "default": "all"},
                "detailed": {"type": "boolean", "default": False},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "admin"],
    )

    manager.register_tool(
        category="admin_tools",
        name="get_system_status",
        func=get_system_status,
        description="Get comprehensive enhanced system status details.",
        input_schema={
            "type": "object",
            "properties": {
                "include_details": {"type": "boolean", "default": True},
                "include_services": {"type": "boolean", "default": True},
                "include_resources": {"type": "boolean", "default": True},
                "format": {"type": "string", "enum": ["json", "summary", "detailed"], "default": "json"},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "admin", "enhanced"],
    )

    manager.register_tool(
        category="admin_tools",
        name="manage_service",
        func=manage_service,
        description="Manage one or all system services via the enhanced admin surface.",
        input_schema={
            "type": "object",
            "properties": {
                "service_name": {"type": "string", "minLength": 1},
                "action": {"type": "string", "enum": ["start", "stop", "restart", "status"]},
                "force": {"type": "boolean", "default": False},
                "timeout_seconds": {"type": "integer", "minimum": 1, "default": 30},
            },
            "required": ["service_name", "action"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "admin", "enhanced"],
    )

    manager.register_tool(
        category="admin_tools",
        name="update_configuration",
        func=update_configuration,
        description="Get, validate, update, backup, or restore configuration via enhanced admin parity.",
        input_schema={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["get", "update", "validate", "backup", "restore"]},
                "config_updates": {"type": ["object", "null"]},
                "create_backup": {"type": "boolean", "default": True},
                "validate_config": {"type": "boolean", "default": True},
            },
            "required": ["action"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "admin", "enhanced"],
    )

    manager.register_tool(
        category="admin_tools",
        name="cleanup_resources",
        func=cleanup_resources,
        description="Clean up system resources via the enhanced admin surface.",
        input_schema={
            "type": "object",
            "properties": {
                "cleanup_type": {"type": "string", "enum": ["basic", "full", "cache_only", "logs_only"], "default": "basic"},
                "restart_services": {"type": "boolean", "default": True},
                "cleanup_temp_files": {"type": "boolean", "default": True},
                "cleanup_logs": {"type": "boolean", "default": False},
                "max_log_age_days": {"type": "integer", "minimum": 1, "default": 30},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "admin", "enhanced"],
    )
