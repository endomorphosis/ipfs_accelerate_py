"""Native rate-limiting tool implementations for unified mcp_server."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _load_rate_limiter_api() -> Dict[str, Any]:
    """Resolve rate limiting engine from source package with compatibility fallback."""
    try:
        from ipfs_datasets_py.rate_limiting.rate_limiting_engine import (  # type: ignore
            RateLimitConfig,
            RateLimitStrategy,
            get_default_rate_limiter,
        )

        return {
            "RateLimitConfig": RateLimitConfig,
            "RateLimitStrategy": RateLimitStrategy,
            "get_default_rate_limiter": get_default_rate_limiter,
        }
    except Exception:
        from ipfs_datasets_py.ipfs_datasets_py.rate_limiting.rate_limiting_engine import (  # type: ignore
            RateLimitConfig,
            RateLimitStrategy,
            get_default_rate_limiter,
        )

        return {
            "RateLimitConfig": RateLimitConfig,
            "RateLimitStrategy": RateLimitStrategy,
            "get_default_rate_limiter": get_default_rate_limiter,
        }


_engine_api = _load_rate_limiter_api()
_rate_limiter = _engine_api["get_default_rate_limiter"]()


async def configure_rate_limits(
    limits: List[Dict[str, Any]],
    apply_immediately: bool = True,
    backup_current: bool = True,
) -> Dict[str, Any]:
    """Configure named rate limits using native unified implementation."""
    _ = apply_immediately

    configured_limits: List[Dict[str, Any]] = []
    errors: List[str] = []

    backup = None
    if backup_current:
        backup = {
            "limits": {
                name: {
                    "strategy": cfg.strategy.value,
                    "requests_per_second": cfg.requests_per_second,
                    "burst_capacity": cfg.burst_capacity,
                    "enabled": cfg.enabled,
                }
                for name, cfg in _rate_limiter.limits.items()
            },
            "backup_time": datetime.now().isoformat(),
        }

    for limit_config in (limits or []):
        try:
            cfg = _engine_api["RateLimitConfig"](
                name=limit_config["name"],
                strategy=_engine_api["RateLimitStrategy"](limit_config.get("strategy", "token_bucket")),
                requests_per_second=float(limit_config["requests_per_second"]),
                burst_capacity=int(limit_config.get("burst_capacity", int(float(limit_config["requests_per_second"]) * 2))),
                window_size_seconds=int(limit_config.get("window_size_seconds", 60)),
                enabled=bool(limit_config.get("enabled", True)),
                penalties=dict(limit_config.get("penalties", {})),
            )
            configured_limits.append(_rate_limiter.configure_limit(cfg))
        except Exception as exc:
            err = f"Failed to configure limit '{limit_config.get('name', 'unknown')}': {exc}"
            logger.error(err)
            errors.append(err)

    return {
        "configured_count": len(configured_limits),
        "configured_limits": configured_limits,
        "errors": errors,
        "applied_immediately": bool(apply_immediately),
        "backup": backup,
        "timestamp": datetime.now().isoformat(),
    }


async def check_rate_limit(
    limit_name: str,
    identifier: str = "default",
    request_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Check whether an identifier is within configured rate limits."""
    result = _rate_limiter.check_rate_limit(str(limit_name), str(identifier))
    payload = dict(result or {})
    payload.update(
        {
            "limit_name": str(limit_name),
            "identifier": str(identifier),
            "check_time": datetime.now().isoformat(),
            "metadata": dict(request_metadata or {}),
        }
    )
    return payload


async def manage_rate_limits(
    action: str,
    limit_name: Optional[str] = None,
    new_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Manage configured rate limits (list/enable/disable/delete/update/stats/reset)."""
    act = str(action or "").strip().lower()
    name = str(limit_name or "").strip()

    if act == "list":
        limits_info = []
        for key, cfg in _rate_limiter.limits.items():
            limits_info.append(
                {
                    "name": key,
                    "strategy": cfg.strategy.value,
                    "requests_per_second": cfg.requests_per_second,
                    "burst_capacity": cfg.burst_capacity,
                    "enabled": cfg.enabled,
                }
            )
        return {
            "action": "list",
            "limits": limits_info,
            "total_count": len(limits_info),
        }

    if act == "enable":
        if not name:
            return {"error": "limit_name required for enable action"}
        if name not in _rate_limiter.limits:
            return {"error": f"Rate limit '{name}' not found"}
        _rate_limiter.limits[name].enabled = True
        return {
            "action": "enable",
            "limit_name": name,
            "enabled": True,
            "timestamp": datetime.now().isoformat(),
        }

    if act == "disable":
        if not name:
            return {"error": "limit_name required for disable action"}
        if name not in _rate_limiter.limits:
            return {"error": f"Rate limit '{name}' not found"}
        _rate_limiter.limits[name].enabled = False
        return {
            "action": "disable",
            "limit_name": name,
            "enabled": False,
            "timestamp": datetime.now().isoformat(),
        }

    if act == "delete":
        if not name:
            return {"error": "limit_name required for delete action"}
        if name not in _rate_limiter.limits:
            return {"error": f"Rate limit '{name}' not found"}
        del _rate_limiter.limits[name]
        _rate_limiter.global_stats["active_limits"] = len(_rate_limiter.limits)
        return {
            "action": "delete",
            "limit_name": name,
            "deleted": True,
            "timestamp": datetime.now().isoformat(),
        }

    if act == "update":
        if not name or not isinstance(new_config, dict):
            return {"error": "limit_name and new_config required for update action"}
        if name not in _rate_limiter.limits:
            return {"error": f"Rate limit '{name}' not found"}

        cfg = _rate_limiter.limits[name]
        if "requests_per_second" in new_config:
            cfg.requests_per_second = float(new_config["requests_per_second"])
        if "burst_capacity" in new_config:
            cfg.burst_capacity = int(new_config["burst_capacity"])
        if "enabled" in new_config:
            cfg.enabled = bool(new_config["enabled"])
        if "strategy" in new_config:
            cfg.strategy = _engine_api["RateLimitStrategy"](new_config["strategy"])

        return {
            "action": "update",
            "limit_name": name,
            "updated_config": {
                "requests_per_second": cfg.requests_per_second,
                "burst_capacity": cfg.burst_capacity,
                "enabled": cfg.enabled,
                "strategy": cfg.strategy.value,
            },
            "timestamp": datetime.now().isoformat(),
        }

    if act == "stats":
        return _rate_limiter.get_stats(name or None)

    if act == "reset":
        return _rate_limiter.reset_limits(name or None)

    return {"error": f"Unknown action: {act}"}


def register_native_rate_limiting_tools(manager: Any) -> None:
    """Register native rate-limiting tools in the unified hierarchical manager."""
    manager.register_tool(
        category="rate_limiting",
        name="configure_rate_limits",
        func=configure_rate_limits,
        description="Configure rate limiting rules using unified native implementation.",
        input_schema={
            "type": "object",
            "properties": {
                "limits": {
                    "type": "array",
                    "items": {"type": "object"},
                },
                "apply_immediately": {"type": "boolean"},
                "backup_current": {"type": "boolean"},
            },
            "required": ["limits"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "rate_limiting"],
    )

    manager.register_tool(
        category="rate_limiting",
        name="check_rate_limit",
        func=check_rate_limit,
        description="Check whether an identifier is within configured rate limits.",
        input_schema={
            "type": "object",
            "properties": {
                "limit_name": {"type": "string"},
                "identifier": {"type": "string"},
                "request_metadata": {"type": ["object", "null"]},
            },
            "required": ["limit_name"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "rate_limiting"],
    )

    manager.register_tool(
        category="rate_limiting",
        name="manage_rate_limits",
        func=manage_rate_limits,
        description="Manage configured rate limits (list, update, enable, disable, delete, stats, reset).",
        input_schema={
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "limit_name": {"type": ["string", "null"]},
                "new_config": {"type": ["object", "null"]},
            },
            "required": ["action"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "rate_limiting"],
    )
