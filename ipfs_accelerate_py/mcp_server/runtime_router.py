"""Dual-runtime router for unified MCP server dispatch.

This module provides a lightweight runtime router that can dispatch tool calls
between standard async execution and Trio-backed execution paths.
"""

from __future__ import annotations

import anyio
import inspect
import logging
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable, Dict, List, Optional

from .exceptions import RuntimeExecutionError, RuntimeNotFoundError, RuntimeRoutingError
from .tool_metadata import ToolMetadataRegistry, get_registry

logger = logging.getLogger(__name__)

RUNTIME_FASTAPI = "fastapi"
RUNTIME_TRIO = "trio"
RUNTIME_AUTO = "auto"
RUNTIME_UNKNOWN = "unknown"

_SUPPORTED_RUNTIMES = {RUNTIME_FASTAPI, RUNTIME_TRIO, RUNTIME_AUTO}


@dataclass
class RuntimeMetrics:
    """Aggregated latency/error metrics for one runtime."""

    request_count: int = 0
    error_count: int = 0
    timeout_count: int = 0
    total_latency_ms: float = 0.0
    min_latency_ms: float = float("inf")
    max_latency_ms: float = 0.0
    latencies: List[float] = field(default_factory=list)

    def record_request(self, latency_ms: float, error: bool = False, timeout: bool = False) -> None:
        """Record one request latency and error state."""
        self.request_count += 1
        if error:
            self.error_count += 1
        if timeout:
            self.timeout_count += 1

        self.total_latency_ms += latency_ms
        self.min_latency_ms = min(self.min_latency_ms, latency_ms)
        self.max_latency_ms = max(self.max_latency_ms, latency_ms)
        self.latencies.append(latency_ms)
        if len(self.latencies) > 1000:
            self.latencies = self.latencies[-1000:]

    @property
    def avg_latency_ms(self) -> float:
        """Return average request latency in milliseconds."""
        if self.request_count == 0:
            return 0.0
        return self.total_latency_ms / self.request_count

    @property
    def p95_latency_ms(self) -> float:
        """Return P95 latency in milliseconds."""
        if not self.latencies:
            return 0.0
        ordered = sorted(self.latencies)
        idx = int(len(ordered) * 0.95)
        return ordered[idx] if idx < len(ordered) else ordered[-1]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metrics for dashboards and diagnostics."""
        return {
            "request_count": self.request_count,
            "error_count": self.error_count,
            "timeout_count": self.timeout_count,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "min_latency_ms": round(self.min_latency_ms, 2) if self.min_latency_ms != float("inf") else 0.0,
            "max_latency_ms": round(self.max_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
        }


class RuntimeRouter:
    """Route tool calls to fastapi/trio execution backends."""

    def __init__(
        self,
        default_runtime: str = RUNTIME_FASTAPI,
        enable_metrics: bool = True,
        trio_bridge_required: bool = False,
        metadata_registry: Optional[ToolMetadataRegistry] = None,
    ) -> None:
        if default_runtime not in _SUPPORTED_RUNTIMES:
            raise RuntimeNotFoundError(default_runtime)

        self.default_runtime = default_runtime
        self.enable_metrics = enable_metrics
        self.trio_bridge_required = bool(trio_bridge_required)
        self._metadata_registry = metadata_registry or get_registry()
        self._tool_runtimes: Dict[str, str] = {}
        self._metrics = {
            RUNTIME_FASTAPI: RuntimeMetrics(),
            RUNTIME_TRIO: RuntimeMetrics(),
            RUNTIME_UNKNOWN: RuntimeMetrics(),
        }
        self._lock = RLock()
        self._is_running = False

    async def startup(self) -> None:
        """Lifecycle hook for runtime initialization."""
        self._is_running = True

    async def shutdown(self) -> None:
        """Lifecycle hook for runtime cleanup."""
        self._is_running = False

    def register_tool_runtime(self, tool_name: str, runtime: str) -> None:
        """Register explicit runtime preference for a tool."""
        if runtime not in _SUPPORTED_RUNTIMES:
            raise RuntimeNotFoundError(runtime)
        self._tool_runtimes[tool_name] = runtime

    def resolve_runtime(self, tool_name: str, tool_func: Optional[Callable[..., Any]] = None) -> str:
        """Resolve runtime from explicit map, metadata attributes, or defaults."""
        runtime = self._tool_runtimes.get(tool_name)
        if runtime:
            return runtime

        metadata = self._metadata_registry.get(tool_name)
        if metadata and metadata.runtime in _SUPPORTED_RUNTIMES:
            return metadata.runtime

        if tool_func is not None:
            # Supports future metadata decorators that attach runtime attributes.
            fn_runtime = getattr(tool_func, "runtime", None) or getattr(tool_func, "__mcp_runtime__", None)
            if isinstance(fn_runtime, str) and fn_runtime in _SUPPORTED_RUNTIMES:
                return fn_runtime

        return self.default_runtime

    def resolve_timeout_seconds(self, tool_name: str, tool_func: Optional[Callable[..., Any]] = None) -> Optional[float]:
        """Resolve per-tool timeout from metadata or function attributes.

        Precedence:
        1. metadata registry entry timeout_seconds
        2. function attribute ``__mcp_timeout_seconds__``
        3. no timeout (None)
        """
        metadata = self._metadata_registry.get(tool_name)
        if metadata and metadata.timeout_seconds is not None:
            return float(metadata.timeout_seconds)

        if tool_func is not None:
            fn_timeout = getattr(tool_func, "__mcp_timeout_seconds__", None)
            if fn_timeout is not None:
                return float(fn_timeout)

        return None

    async def route_tool_call(
        self,
        registered_tool_name: str,
        tool_func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute a tool via the resolved runtime backend."""
        if not callable(tool_func):
            raise RuntimeRoutingError(f"Tool function for '{registered_tool_name}' is not callable")

        runtime = self.resolve_runtime(registered_tool_name, tool_func)
        timeout_seconds = self.resolve_timeout_seconds(registered_tool_name, tool_func)
        start = time.perf_counter()
        error = False
        timed_out = False

        try:
            if runtime == RUNTIME_AUTO:
                runtime = self.default_runtime

            async def _invoke() -> Any:
                if runtime == RUNTIME_TRIO:
                    return await self._execute_trio(tool_func, *args, **kwargs)
                if runtime == RUNTIME_FASTAPI:
                    return await self._execute_fastapi(tool_func, *args, **kwargs)
                raise RuntimeNotFoundError(runtime)

            if timeout_seconds is not None and timeout_seconds > 0:
                with anyio.fail_after(timeout_seconds):
                    result = await _invoke()
            else:
                result = await _invoke()

            return result
        except TimeoutError as exc:
            error = True
            timed_out = True
            raise RuntimeExecutionError(runtime, registered_tool_name, f"timed out after {timeout_seconds}s") from exc
        except Exception as exc:
            error = True
            if isinstance(exc, RuntimeRoutingError):
                raise
            raise RuntimeExecutionError(runtime, registered_tool_name, exc) from exc
        finally:
            if self.enable_metrics:
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                with self._lock:
                    bucket = self._metrics.get(runtime, self._metrics[RUNTIME_UNKNOWN])
                    bucket.record_request(elapsed_ms, error=error, timeout=timed_out)

    async def _execute_fastapi(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute in standard async context (asyncio/anyio path)."""
        result = func(*args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    async def _execute_trio(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute via Trio bridge when available."""
        try:
            from ipfs_accelerate_py.mcplusplus_module.trio.bridge import run_in_trio

            return await run_in_trio(func, *args, **kwargs)
        except ImportError:
            if self.trio_bridge_required:
                raise RuntimeRoutingError(
                    "Trio bridge unavailable and trio_bridge_required=True"
                )
            # Backward-compatible fallback keeps routing functional in environments
            # without trio extras.
            logger.warning("Trio bridge unavailable; falling back to standard execution")
            return await self._execute_fastapi(func, *args, **kwargs)

    def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Return collected runtime metrics."""
        with self._lock:
            return {runtime: metric.to_dict() for runtime, metric in self._metrics.items()}


__all__ = [
    "RuntimeMetrics",
    "RuntimeRouter",
    "RUNTIME_FASTAPI",
    "RUNTIME_TRIO",
    "RUNTIME_AUTO",
    "RUNTIME_UNKNOWN",
]
