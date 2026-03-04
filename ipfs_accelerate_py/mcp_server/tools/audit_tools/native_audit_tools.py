"""Native audit tool implementations for unified mcp_server."""

from __future__ import annotations

from datetime import datetime
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


def _is_iso8601(value: str) -> bool:
    """Return True when value is parseable as ISO-8601 datetime."""
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return True
    except Exception:
        return False


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
    normalized_action = str(action or "").strip()
    if not normalized_action:
        return {
            "status": "error",
            "message": "action is required",
            "action": action,
        }

    normalized_severity = str(severity or "").strip().lower()
    valid_severities = {"info", "warning", "error", "critical", "debug"}
    if normalized_severity not in valid_severities:
        return {
            "status": "error",
            "message": "severity must be one of: info, warning, error, critical, debug",
            "severity": severity,
        }

    normalized_resource_id = str(resource_id).strip() if resource_id is not None else None
    if resource_id is not None and not normalized_resource_id:
        return {
            "status": "error",
            "message": "resource_id must be a non-empty string when provided",
            "resource_id": resource_id,
        }
    normalized_resource_type = str(resource_type).strip() if resource_type is not None else None
    if resource_type is not None and not normalized_resource_type:
        return {
            "status": "error",
            "message": "resource_type must be a non-empty string when provided",
            "resource_type": resource_type,
        }
    normalized_user_id = str(user_id).strip() if user_id is not None else None
    if user_id is not None and not normalized_user_id:
        return {
            "status": "error",
            "message": "user_id must be a non-empty string when provided",
            "user_id": user_id,
        }
    if details is not None and not isinstance(details, dict):
        return {
            "status": "error",
            "message": "details must be an object when provided",
            "details": details,
        }

    normalized_source_ip = str(source_ip).strip() if source_ip is not None else None
    if source_ip is not None and not normalized_source_ip:
        return {
            "status": "error",
            "message": "source_ip must be a non-empty string when provided",
            "source_ip": source_ip,
        }

    normalized_tags: Optional[List[str]] = None
    if tags is not None:
        if not isinstance(tags, list) or not all(isinstance(item, str) for item in tags):
            return {
                "status": "error",
                "message": "tags must be an array of strings when provided",
                "tags": tags,
            }
        normalized_tags = [str(item).strip() for item in tags]
        if any(not item for item in normalized_tags):
            return {
                "status": "error",
                "message": "tags cannot contain empty strings",
                "tags": tags,
            }

    result = await _API["record_audit_event"](
        action=normalized_action,
        resource_id=normalized_resource_id,
        resource_type=normalized_resource_type,
        user_id=normalized_user_id,
        details=details,
        source_ip=normalized_source_ip,
        severity=normalized_severity,
        tags=normalized_tags,
    )
    payload = dict(result or {})
    if "error" in payload and payload.get("error"):
        payload.setdefault("status", "error")
    else:
        payload.setdefault("status", "success")
    payload.setdefault("action", normalized_action)
    payload.setdefault("severity", normalized_severity)
    return payload


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
    normalized_report_type = str(report_type or "").strip().lower()
    valid_report_types = {"security", "compliance", "operational", "comprehensive"}
    if normalized_report_type not in valid_report_types:
        return {
            "status": "error",
            "message": "report_type must be one of: security, compliance, operational, comprehensive",
            "report_type": report_type,
        }

    normalized_start = str(start_time).strip() if start_time is not None else None
    if start_time is not None and (not normalized_start or not _is_iso8601(normalized_start)):
        return {
            "status": "error",
            "message": "start_time must be a valid ISO-8601 datetime when provided",
            "start_time": start_time,
        }
    normalized_end = str(end_time).strip() if end_time is not None else None
    if end_time is not None and (not normalized_end or not _is_iso8601(normalized_end)):
        return {
            "status": "error",
            "message": "end_time must be a valid ISO-8601 datetime when provided",
            "end_time": end_time,
        }

    if filters is not None and not isinstance(filters, dict):
        return {
            "status": "error",
            "message": "filters must be an object when provided",
            "filters": filters,
        }

    normalized_output_format = str(output_format or "").strip().lower()
    valid_output_formats = {"json", "html", "pdf"}
    if normalized_output_format not in valid_output_formats:
        return {
            "status": "error",
            "message": "output_format must be one of: json, html, pdf",
            "output_format": output_format,
        }

    normalized_output_path = str(output_path).strip() if output_path is not None else None
    if output_path is not None and not normalized_output_path:
        return {
            "status": "error",
            "message": "output_path must be a non-empty string when provided",
            "output_path": output_path,
        }

    if not isinstance(include_details, bool):
        return {
            "status": "error",
            "message": "include_details must be a boolean",
            "include_details": include_details,
        }

    result = await _API["generate_audit_report"](
        report_type=normalized_report_type,
        start_time=normalized_start,
        end_time=normalized_end,
        filters=filters,
        output_format=normalized_output_format,
        output_path=normalized_output_path,
        include_details=include_details,
    )
    payload = dict(result or {})
    if "error" in payload and payload.get("error"):
        payload.setdefault("status", "error")
    else:
        payload.setdefault("status", "success")
    payload.setdefault("report_type", normalized_report_type)
    payload.setdefault("output_format", normalized_output_format)
    return payload


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
                "severity": {
                    "type": "string",
                    "enum": ["info", "warning", "error", "critical", "debug"],
                    "default": "info",
                },
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
                "report_type": {
                    "type": "string",
                    "enum": ["security", "compliance", "operational", "comprehensive"],
                    "default": "comprehensive",
                },
                "start_time": {"type": ["string", "null"], "format": "date-time"},
                "end_time": {"type": ["string", "null"], "format": "date-time"},
                "filters": {"type": ["object", "null"]},
                "output_format": {
                    "type": "string",
                    "enum": ["json", "html", "pdf"],
                    "default": "json",
                },
                "output_path": {"type": ["string", "null"]},
                "include_details": {"type": "boolean", "default": True},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "audit"],
    )
