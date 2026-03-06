"""Native rate-limiting tool implementations for unified mcp_server."""

from __future__ import annotations

import logging
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _error_result(message: str, **extra: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"status": "error", "error": message}
    payload.update(extra)
    return payload


def _load_rate_limiter_api() -> Dict[str, Any]:
    """Resolve rate limiting engine from source package with compatibility fallback."""
    candidates = [
        "ipfs_datasets_py.rate_limiting.rate_limiting_engine",
        "ipfs_datasets_py.ipfs_datasets_py.rate_limiting.rate_limiting_engine",
    ]

    for mod_name in candidates:
        try:
            mod = __import__(mod_name, fromlist=["RateLimitConfig", "RateLimitStrategy", "get_default_rate_limiter"])
            return {
                "RateLimitConfig": getattr(mod, "RateLimitConfig"),
                "RateLimitStrategy": getattr(mod, "RateLimitStrategy"),
                "get_default_rate_limiter": getattr(mod, "get_default_rate_limiter"),
            }
        except Exception:
            continue

    logger.warning("Rate limiting engine unavailable; using in-process fallback implementation")

    class RateLimitStrategy(Enum):
        token_bucket = "token_bucket"

    @dataclass
    class RateLimitConfig:
        name: str
        strategy: RateLimitStrategy = RateLimitStrategy.token_bucket
        requests_per_second: float = 10.0
        burst_capacity: int = 20
        window_size_seconds: int = 60
        enabled: bool = True
        penalties: Dict[str, Any] | None = None

    class _FallbackLimiter:
        def __init__(self) -> None:
            self.limits: Dict[str, RateLimitConfig] = {}
            self.global_stats: Dict[str, Any] = {"active_limits": 0}

        def configure_limit(self, cfg: RateLimitConfig) -> Dict[str, Any]:
            self.limits[str(cfg.name)] = cfg
            self.global_stats["active_limits"] = len(self.limits)
            return {
                "name": str(cfg.name),
                "strategy": str(getattr(cfg.strategy, "value", cfg.strategy)),
                "requests_per_second": float(cfg.requests_per_second),
                "burst_capacity": int(cfg.burst_capacity),
                "enabled": bool(cfg.enabled),
            }

        def check_rate_limit(self, name: str, identifier: str) -> Dict[str, Any]:
            cfg = self.limits.get(str(name))
            if cfg is not None and not bool(cfg.enabled):
                return {"allowed": True, "reason": "limit_disabled", "remaining": None, "retry_after_s": 0}
            return {"allowed": True, "reason": "fallback_allow", "remaining": None, "retry_after_s": 0}

        def get_stats(self, name: str | None = None) -> Dict[str, Any]:
            if name:
                cfg = self.limits.get(str(name))
                if cfg is None:
                    return {"error": f"Rate limit '{name}' not found"}
                return {
                    "name": str(name),
                    "strategy": str(getattr(cfg.strategy, "value", cfg.strategy)),
                    "requests_per_second": float(cfg.requests_per_second),
                    "burst_capacity": int(cfg.burst_capacity),
                    "enabled": bool(cfg.enabled),
                }
            return {
                "active_limits": len(self.limits),
                "limits": {
                    k: {
                        "strategy": str(getattr(v.strategy, "value", v.strategy)),
                        "requests_per_second": float(v.requests_per_second),
                        "burst_capacity": int(v.burst_capacity),
                        "enabled": bool(v.enabled),
                    }
                    for k, v in self.limits.items()
                },
            }

        def reset_limits(self, name: str | None = None) -> Dict[str, Any]:
            if name:
                if str(name) not in self.limits:
                    return {"error": f"Rate limit '{name}' not found"}
                return {"reset": True, "limit_name": str(name)}
            return {"reset": True, "limit_name": None}

    _fallback_singleton = _FallbackLimiter()

    def get_default_rate_limiter() -> _FallbackLimiter:
        return _fallback_singleton

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
    if not isinstance(limits, list):
        return _error_result("limits must be a list", configured_count=0, configured_limits=[], errors=[])
    if not isinstance(apply_immediately, bool):
        return _error_result("apply_immediately must be a boolean")
    if not isinstance(backup_current, bool):
        return _error_result("backup_current must be a boolean")

    _ = apply_immediately

    configured_limits: List[Dict[str, Any]] = []
    errors: List[str] = []

    backup = None
    if backup_current:
        backup = {
            "limits": {
                name: {
                    "strategy": str(getattr(cfg.strategy, "value", cfg.strategy)),
                    "requests_per_second": cfg.requests_per_second,
                    "burst_capacity": cfg.burst_capacity,
                    "enabled": cfg.enabled,
                }
                for name, cfg in _rate_limiter.limits.items()
            },
            "backup_time": datetime.now().isoformat(),
        }

    for index, limit_config in enumerate(limits):
        if not isinstance(limit_config, dict):
            errors.append(f"limits[{index}] must be an object")
            continue
        limit_name = str(limit_config.get("name", "")).strip()
        if not limit_name:
            errors.append(f"limits[{index}].name must be a non-empty string")
            continue
        if "requests_per_second" not in limit_config:
            errors.append(f"limits[{index}].requests_per_second is required")
            continue

        try:
            requests_per_second = float(limit_config["requests_per_second"])
            if requests_per_second <= 0:
                raise ValueError("requests_per_second must be > 0")

            burst_capacity = int(
                limit_config.get("burst_capacity", int(float(limit_config["requests_per_second"]) * 2))
            )
            if burst_capacity <= 0:
                raise ValueError("burst_capacity must be > 0")

            window_size_seconds = int(limit_config.get("window_size_seconds", 60))
            if window_size_seconds <= 0:
                raise ValueError("window_size_seconds must be > 0")

            cfg = _engine_api["RateLimitConfig"](
                name=limit_name,
                strategy=_engine_api["RateLimitStrategy"](limit_config.get("strategy", "token_bucket")),
                requests_per_second=requests_per_second,
                burst_capacity=burst_capacity,
                window_size_seconds=window_size_seconds,
                enabled=bool(limit_config.get("enabled", True)),
                penalties=dict(limit_config.get("penalties", {})),
            )
            configured_limits.append(_rate_limiter.configure_limit(cfg))
        except Exception as exc:
            err = f"Failed to configure limit '{limit_name or 'unknown'}': {exc}"
            logger.error(err)
            errors.append(err)

    return {
        "status": "error" if errors else "success",
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
    normalized_limit_name = str(limit_name or "").strip()
    if not normalized_limit_name:
        return _error_result("limit_name must be a non-empty string", allowed=False)

    normalized_identifier = str(identifier or "").strip()
    if not normalized_identifier:
        return _error_result(
            "identifier must be a non-empty string",
            allowed=False,
            limit_name=normalized_limit_name,
            identifier=str(identifier or ""),
        )

    if request_metadata is not None and not isinstance(request_metadata, dict):
        return _error_result(
            "request_metadata must be an object or null",
            allowed=False,
            limit_name=normalized_limit_name,
            identifier=normalized_identifier,
        )

    try:
        result = _rate_limiter.check_rate_limit(normalized_limit_name, normalized_identifier)
    except Exception as exc:
        return _error_result(
            f"check_rate_limit failed: {exc}",
            allowed=False,
            limit_name=normalized_limit_name,
            identifier=normalized_identifier,
        )

    payload = dict(result or {})
    payload.update(
        {
            "status": payload.get("status", "success" if payload.get("allowed", True) else "error"),
            "limit_name": normalized_limit_name,
            "identifier": normalized_identifier,
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
    if not act:
        return _error_result(
            "action is required",
            valid_actions=["list", "enable", "disable", "delete", "update", "stats", "reset"],
        )

    valid_actions = {"list", "enable", "disable", "delete", "update", "stats", "reset"}
    if act not in valid_actions:
        return _error_result(
            f"Unknown action: {act}",
            valid_actions=["list", "enable", "disable", "delete", "update", "stats", "reset"],
        )

    name = str(limit_name or "").strip()

    if act in {"stats", "reset"} and limit_name is not None and not name:
        return _error_result(f"limit_name must be a non-empty string when provided for {act}")

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
            "status": "success",
            "action": "list",
            "limits": limits_info,
            "total_count": len(limits_info),
        }

    if act == "enable":
        if not name:
            return _error_result("limit_name required for enable action")
        if name not in _rate_limiter.limits:
            return _error_result(f"Rate limit '{name}' not found")
        _rate_limiter.limits[name].enabled = True
        return {
            "status": "success",
            "action": "enable",
            "limit_name": name,
            "enabled": True,
            "timestamp": datetime.now().isoformat(),
        }

    if act == "disable":
        if not name:
            return _error_result("limit_name required for disable action")
        if name not in _rate_limiter.limits:
            return _error_result(f"Rate limit '{name}' not found")
        _rate_limiter.limits[name].enabled = False
        return {
            "status": "success",
            "action": "disable",
            "limit_name": name,
            "enabled": False,
            "timestamp": datetime.now().isoformat(),
        }

    if act == "delete":
        if not name:
            return _error_result("limit_name required for delete action")
        if name not in _rate_limiter.limits:
            return _error_result(f"Rate limit '{name}' not found")
        del _rate_limiter.limits[name]
        _rate_limiter.global_stats["active_limits"] = len(_rate_limiter.limits)
        return {
            "status": "success",
            "action": "delete",
            "limit_name": name,
            "deleted": True,
            "timestamp": datetime.now().isoformat(),
        }

    if act == "update":
        if not name or not isinstance(new_config, dict):
            return _error_result("limit_name and new_config required for update action")
        if name not in _rate_limiter.limits:
            return _error_result(f"Rate limit '{name}' not found")

        cfg = _rate_limiter.limits[name]
        if "requests_per_second" in new_config:
            new_rps = float(new_config["requests_per_second"])
            if new_rps <= 0:
                return _error_result("requests_per_second must be > 0")
            cfg.requests_per_second = new_rps
        if "burst_capacity" in new_config:
            new_burst = int(new_config["burst_capacity"])
            if new_burst <= 0:
                return _error_result("burst_capacity must be > 0")
            cfg.burst_capacity = new_burst
        if "enabled" in new_config:
            if not isinstance(new_config["enabled"], bool):
                return _error_result("enabled must be a boolean")
            cfg.enabled = bool(new_config["enabled"])
        if "strategy" in new_config:
            try:
                cfg.strategy = _engine_api["RateLimitStrategy"](new_config["strategy"])
            except Exception:
                return _error_result("strategy must be a valid rate limiting strategy")

        return {
            "status": "success",
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
        payload = dict(_rate_limiter.get_stats(name or None) or {})
        if "error" in payload:
            payload.setdefault("status", "error")
        else:
            payload.setdefault("status", "success")
        return payload

    if act == "reset":
        payload = dict(_rate_limiter.reset_limits(name or None) or {})
        if "error" in payload:
            payload.setdefault("status", "error")
        else:
            payload.setdefault("status", "success")
        return payload

    return _error_result(f"Unknown action: {act}")


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
                "apply_immediately": {"type": "boolean", "default": True},
                "backup_current": {"type": "boolean", "default": True},
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
                "limit_name": {"type": "string", "minLength": 1},
                "identifier": {"type": "string", "minLength": 1, "default": "default"},
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
                "action": {
                    "type": "string",
                    "enum": ["list", "enable", "disable", "delete", "update", "stats", "reset"],
                },
                "limit_name": {"type": ["string", "null"], "minLength": 1},
                "new_config": {"type": ["object", "null"]},
            },
            "required": ["action"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "rate_limiting"],
    )
