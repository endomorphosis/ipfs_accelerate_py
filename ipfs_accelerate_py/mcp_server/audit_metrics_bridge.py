"""Policy-audit to metrics bridge for unified MCP runtime.

Connects :class:`PolicyAuditLog` decision entries to
:class:`PrometheusExporter.record_tool_call` for lightweight policy observability.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class AuditMetricsBridge:
    """Forward policy audit entries to Prometheus tool-call counters."""

    def __init__(self, audit_log: Any, exporter: Any, *, category: str = "policy") -> None:
        self._audit = audit_log
        self._exporter = exporter
        self._category = str(category or "policy")
        self._attached = False
        self._forwarded_count = 0
        self._previous_sink: Optional[Callable[[Any], None]] = None

    def _sink(self, entry: Any) -> None:
        try:
            decision = str(getattr(entry, "decision", "unknown") or "unknown")
            tool = str(getattr(entry, "tool", "") or "unknown")
            status = "allowed" if decision in ("allow", "allow_with_obligations") else "denied"
            self._exporter.record_tool_call(
                category=self._category,
                tool=tool,
                status=status,
                latency_seconds=0.0,
            )
            self._forwarded_count += 1
        except Exception as exc:  # pragma: no cover
            logger.warning("AuditMetricsBridge sink error: %s", exc)
        finally:
            if self._previous_sink is not None:
                try:
                    self._previous_sink(entry)
                except Exception as exc:  # pragma: no cover
                    logger.warning("AuditMetricsBridge previous sink error: %s", exc)

    def attach(self) -> None:
        if self._attached:
            return
        self._previous_sink = getattr(self._audit, "_sink", None)
        self._audit._sink = self._sink
        self._attached = True

    def detach(self) -> None:
        if not self._attached:
            return
        self._audit._sink = self._previous_sink
        self._attached = False

    @property
    def is_attached(self) -> bool:
        return self._attached

    @property
    def forwarded_count(self) -> int:
        return self._forwarded_count

    def get_info(self) -> dict[str, Any]:
        return {
            "attached": self._attached,
            "category": self._category,
            "forwarded_count": self._forwarded_count,
        }


def connect_audit_to_prometheus(audit_log: Any, exporter: Any, *, category: str = "policy") -> AuditMetricsBridge:
    """Create and attach an :class:`AuditMetricsBridge`."""
    bridge = AuditMetricsBridge(audit_log, exporter, category=category)
    bridge.attach()
    return bridge


__all__ = ["AuditMetricsBridge", "connect_audit_to_prometheus"]
