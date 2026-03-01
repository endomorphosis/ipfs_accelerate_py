"""OpenTelemetry tracing compatibility helpers for unified MCP runtime."""

from __future__ import annotations

import functools
import logging
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

logger = logging.getLogger(__name__)

try:
    from opentelemetry import trace as otel_trace  # type: ignore[import]
    from opentelemetry.sdk.trace import TracerProvider  # type: ignore[import]
    from opentelemetry.sdk.trace.export import BatchSpanProcessor  # type: ignore[import]

    OTEL_AVAILABLE = True
except ImportError:
    otel_trace = None  # type: ignore[assignment]
    TracerProvider = None  # type: ignore[assignment]
    BatchSpanProcessor = None  # type: ignore[assignment]
    OTEL_AVAILABLE = False

try:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter  # type: ignore[import]

    OTLP_GRPC_AVAILABLE = True
except ImportError:
    OTLPSpanExporter = None  # type: ignore[assignment]
    OTLP_GRPC_AVAILABLE = False

try:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as OTLPHTTPSpanExporter  # type: ignore[import]

    OTLP_HTTP_AVAILABLE = True
except ImportError:
    OTLPHTTPSpanExporter = None  # type: ignore[assignment]
    OTLP_HTTP_AVAILABLE = False


class _NoOpSpan:
    """Inert span object used when OTel is unavailable."""

    def set_attribute(self, key: str, value: Any) -> None:
        _ = (key, value)

    def set_status(self, status: Any, description: str = "") -> None:
        _ = (status, description)

    def record_exception(self, exc: Exception) -> None:
        _ = exc

    def end(self) -> None:
        return

    def __enter__(self) -> "_NoOpSpan":
        return self

    def __exit__(self, *_args: Any) -> None:
        return


_NOOP_SPAN = _NoOpSpan()
_tracer_provider: Optional[Any] = None


def _build_exporter(endpoint: str, protocol: str, insecure: bool) -> Optional[Any]:
    if protocol == "grpc" and OTLP_GRPC_AVAILABLE:
        return OTLPSpanExporter(endpoint=endpoint, insecure=insecure)
    if protocol == "http" and OTLP_HTTP_AVAILABLE:
        return OTLPHTTPSpanExporter(endpoint=endpoint)
    logger.warning("OTLP exporter for protocol=%r is unavailable", protocol)
    return None


def configure_tracing(
    service_name: str = "ipfs-mcp-server",
    otlp_endpoint: Optional[str] = None,
    *,
    export_protocol: str = "grpc",
    insecure: bool = True,
) -> bool:
    """Configure global OTel tracing provider when dependencies are available."""
    global _tracer_provider

    if not OTEL_AVAILABLE:
        logger.warning("opentelemetry packages unavailable; tracing disabled")
        return False

    from opentelemetry.sdk.resources import Resource  # type: ignore[import]

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    if otlp_endpoint:
        exporter = _build_exporter(otlp_endpoint, export_protocol, insecure)
        if exporter is not None:
            provider.add_span_processor(BatchSpanProcessor(exporter))

    otel_trace.set_tracer_provider(provider)
    _tracer_provider = provider
    return True


class MCPTracer:
    """Tracing helper for MCP dispatch instrumentation."""

    def __init__(self, tracer_name: str = "ipfs_accelerate_py.mcp_server") -> None:
        self.tracer_name = tracer_name
        self._tracer: Optional[Any] = None

    def _get_tracer(self) -> Any:
        if not OTEL_AVAILABLE:
            return None
        if self._tracer is None:
            self._tracer = otel_trace.get_tracer(self.tracer_name)
        return self._tracer

    @contextmanager
    def start_dispatch_span(
        self,
        category: str,
        tool: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Generator[Any, None, None]:
        tracer = self._get_tracer()
        if tracer is None:
            yield _NOOP_SPAN
            return

        with tracer.start_as_current_span(f"mcp.dispatch/{category}/{tool}") as span:
            span.set_attribute("mcp.category", category)
            span.set_attribute("mcp.tool", tool)
            if params:
                span.set_attribute("mcp.params_keys", ",".join(sorted(params.keys())))
            try:
                yield span
            except BaseException as exc:
                span.record_exception(exc)
                if OTEL_AVAILABLE:
                    span.set_status(otel_trace.StatusCode.ERROR, description=str(exc))  # type: ignore[attr-defined]
                raise

    def set_span_ok(self, span: Any, result: Any = None) -> None:
        if OTEL_AVAILABLE and not isinstance(span, _NoOpSpan):
            span.set_status(otel_trace.StatusCode.OK)  # type: ignore[attr-defined]
            if isinstance(result, dict) and "status" in result:
                span.set_attribute("mcp.result.status", str(result["status"]))

    def trace_tool_call(self, func: Any) -> Any:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            category = args[0] if len(args) > 0 else kwargs.get("category", "unknown")
            tool = args[1] if len(args) > 1 else kwargs.get("tool", "unknown")
            params = args[2] if len(args) > 2 else kwargs.get("params", {})
            with self.start_dispatch_span(str(category), str(tool), params) as span:
                result = await func(*args, **kwargs)
                self.set_span_ok(span, result)
                return result

        return wrapper

    def get_info(self) -> Dict[str, Any]:
        return {
            "tracer": "opentelemetry",
            "tracer_name": self.tracer_name,
            "otel_available": OTEL_AVAILABLE,
            "otlp_grpc_available": OTLP_GRPC_AVAILABLE,
            "otlp_http_available": OTLP_HTTP_AVAILABLE,
        }


__all__ = [
    "MCPTracer",
    "configure_tracing",
    "OTEL_AVAILABLE",
    "OTLP_GRPC_AVAILABLE",
    "OTLP_HTTP_AVAILABLE",
]
