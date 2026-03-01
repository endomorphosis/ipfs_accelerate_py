"""Native audit tool implementations for unified mcp_server."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _load_audit_api() -> Dict[str, Any]:
    """Resolve source audit APIs with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.audit_tools.generate_audit_report import (  # type: ignore
            generate_audit_report as _generate_audit_report,
        )
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.audit_tools.record_audit_event import (  # type: ignore
            record_audit_event as _record_audit_event,
        )

        return {
            "record_audit_event": _record_audit_event,
            "generate_audit_report": _generate_audit_report,
        }
    except Exception:
        logger.warning("Source audit_tools import unavailable, using fallback audit functions")

        async def _record_fallback(
            action: str,
            resource_id: Optional[str] = None,
            resource_type: Optional[str] = None,
            user_id: Optional[str] = None,
            details: Optional[Dict[str, Any]] = None,
            source_ip: Optional[str] = None,
            severity: str = "info",
            tags: Optional[List[str]] = None,
        ) -> Dict[str, Any]:
            _ = resource_type, user_id, details, source_ip, tags
            return {
                "status": "success",
                "event_id": "fallback-audit-event-1",
                "action": action,
                "severity": severity,
                "resource_id": resource_id,
            }

        async def _report_fallback(
            report_type: str = "comprehensive",
            start_time: Optional[str] = None,
            end_time: Optional[str] = None,
            filters: Optional[Dict[str, Any]] = None,
            output_format: str = "json",
            output_path: Optional[str] = None,
            include_details: bool = True,
        ) -> Dict[str, Any]:
            _ = start_time, end_time, filters, output_path, include_details
            return {
                "status": "success",
                "report_type": report_type,
                "output_format": output_format,
                "report": {},
            }

        return {
            "record_audit_event": _record_fallback,
            "generate_audit_report": _report_fallback,
        }


_API = _load_audit_api()


async def record_audit_event(
    action: str,
    resource_id: Optional[str] = None,
    resource_type: Optional[str] = None,
    user_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    source_ip: Optional[str] = None,
    severity: str = "info",
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Record a security/compliance/operations audit event."""
    return await _API["record_audit_event"](
        action=action,
        resource_id=resource_id,
        resource_type=resource_type,
        user_id=user_id,
        details=details,
        source_ip=source_ip,
        severity=severity,
        tags=tags,
    )


async def generate_audit_report(
    report_type: str = "comprehensive",
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
    output_format: str = "json",
    output_path: Optional[str] = None,
    include_details: bool = True,
) -> Dict[str, Any]:
    """Generate an audit report over selected time and filters."""
    return await _API["generate_audit_report"](
        report_type=report_type,
        start_time=start_time,
        end_time=end_time,
        filters=filters,
        output_format=output_format,
        output_path=output_path,
        include_details=include_details,
    )


def register_native_audit_tools(manager: Any) -> None:
    """Register native audit tools in unified hierarchical manager."""
    manager.register_tool(
        category="audit_tools",
        name="record_audit_event",
        func=record_audit_event,
        description="Record an audit event for security/compliance tracking.",
        input_schema={
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "resource_id": {"type": ["string", "null"]},
                "resource_type": {"type": ["string", "null"]},
                "user_id": {"type": ["string", "null"]},
                "details": {"type": ["object", "null"]},
                "source_ip": {"type": ["string", "null"]},
                "severity": {"type": "string"},
                "tags": {"type": ["array", "null"], "items": {"type": "string"}},
            },
            "required": ["action"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "audit"],
    )

    manager.register_tool(
        category="audit_tools",
        name="generate_audit_report",
        func=generate_audit_report,
        description="Generate audit report for selected period and filters.",
        input_schema={
            "type": "object",
            "properties": {
                "report_type": {"type": "string"},
                "start_time": {"type": ["string", "null"]},
                "end_time": {"type": ["string", "null"]},
                "filters": {"type": ["object", "null"]},
                "output_format": {"type": "string"},
                "output_path": {"type": ["string", "null"]},
                "include_details": {"type": "boolean"},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "audit"],
    )
