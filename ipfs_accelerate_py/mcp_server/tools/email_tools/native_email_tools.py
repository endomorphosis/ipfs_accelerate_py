"""Native email tool implementations for unified mcp_server."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _load_email_api() -> Dict[str, Any]:
    """Resolve source email APIs with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.email_tools.email_analyze import (  # type: ignore
            email_analyze_export as _email_analyze_export,
            email_search_export as _email_search_export,
        )
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.email_tools.email_connect import (  # type: ignore
            email_list_folders as _email_list_folders,
            email_test_connection as _email_test_connection,
        )
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.email_tools.email_export import (  # type: ignore
            email_parse_eml as _email_parse_eml,
        )

        return {
            "email_test_connection": _email_test_connection,
            "email_list_folders": _email_list_folders,
            "email_analyze_export": _email_analyze_export,
            "email_search_export": _email_search_export,
            "email_parse_eml": _email_parse_eml,
        }
    except Exception:
        logger.warning("Source email_tools import unavailable, using fallback email functions")

        async def _test_conn_fallback(
            protocol: str = "imap",
            server: Optional[str] = None,
            port: Optional[int] = None,
            username: Optional[str] = None,
            password: Optional[str] = None,
            use_ssl: bool = True,
            timeout: int = 30,
        ) -> Dict[str, Any]:
            _ = port, username, password, use_ssl, timeout
            if not server:
                return {
                    "status": "error",
                    "error": "server is required",
                    "protocol": protocol,
                }
            return {
                "status": "error",
                "protocol": protocol,
                "server": server,
                "connected": False,
                "error": "Email processor not available",
            }

        async def _list_folders_fallback(
            server: Optional[str] = None,
            port: Optional[int] = None,
            username: Optional[str] = None,
            password: Optional[str] = None,
            use_ssl: bool = True,
            timeout: int = 30,
        ) -> Dict[str, Any]:
            _ = port, username, password, use_ssl, timeout
            if not server:
                return {
                    "status": "error",
                    "error": "server is required",
                }
            return {
                "status": "success",
                "folder_count": 0,
                "folders": [],
                "server": server,
            }

        async def _analyze_export_fallback(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            _ = args, kwargs
            return {
                "status": "error",
                "error": "Email analysis backend not available",
            }

        async def _search_export_fallback(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            _ = args, kwargs
            return {
                "status": "error",
                "error": "Email analysis backend not available",
            }

        async def _parse_eml_fallback(
            file_path: str,
            include_attachments: bool = True,
        ) -> Dict[str, Any]:
            _ = include_attachments
            return {
                "status": "error",
                "error": "Email processor not available",
                "file_path": file_path,
            }

        return {
            "email_test_connection": _test_conn_fallback,
            "email_list_folders": _list_folders_fallback,
            "email_analyze_export": _analyze_export_fallback,
            "email_search_export": _search_export_fallback,
            "email_parse_eml": _parse_eml_fallback,
        }


_API = _load_email_api()


async def email_test_connection(
    protocol: str = "imap",
    server: Optional[str] = None,
    port: Optional[int] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    use_ssl: bool = True,
    timeout: int = 30,
) -> Dict[str, Any]:
    """Test IMAP/POP3 connectivity."""
    normalized_protocol = str(protocol or "").strip().lower()
    if normalized_protocol not in {"imap", "pop3"}:
        return {
            "status": "error",
            "message": "protocol must be one of: imap, pop3",
            "protocol": protocol,
        }

    normalized_server = str(server).strip() if server is not None else None
    if server is not None and not normalized_server:
        return {
            "status": "error",
            "message": "server must be a non-empty string when provided",
            "server": server,
        }
    normalized_username = str(username).strip() if username is not None else None
    if username is not None and not normalized_username:
        return {
            "status": "error",
            "message": "username must be a non-empty string when provided",
            "username": username,
        }
    normalized_password = str(password).strip() if password is not None else None
    if password is not None and not normalized_password:
        return {
            "status": "error",
            "message": "password must be a non-empty string when provided",
            "password": "***",
        }
    if port is not None and (not isinstance(port, int) or port <= 0):
        return {
            "status": "error",
            "message": "port must be a positive integer when provided",
            "port": port,
        }
    if not isinstance(use_ssl, bool):
        return {
            "status": "error",
            "message": "use_ssl must be a boolean",
            "use_ssl": use_ssl,
        }
    if not isinstance(timeout, int) or timeout < 1:
        return {
            "status": "error",
            "message": "timeout must be an integer >= 1",
            "timeout": timeout,
        }

    result = await _API["email_test_connection"](
        protocol=normalized_protocol,
        server=normalized_server,
        port=port,
        username=normalized_username,
        password=normalized_password,
        use_ssl=use_ssl,
        timeout=timeout,
    )
    payload = dict(result or {})
    if "error" in payload and payload.get("error"):
        payload.setdefault("status", "error")
    else:
        payload.setdefault("status", "success")
    payload.setdefault("protocol", normalized_protocol)
    payload.setdefault("use_ssl", use_ssl)
    payload.setdefault("timeout", timeout)
    if normalized_server is not None:
        payload.setdefault("server", normalized_server)
    return payload


async def email_list_folders(
    server: Optional[str] = None,
    port: Optional[int] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    use_ssl: bool = True,
    timeout: int = 30,
) -> Dict[str, Any]:
    """List available mailbox folders from IMAP server."""
    normalized_server = str(server).strip() if server is not None else None
    if server is not None and not normalized_server:
        return {
            "status": "error",
            "message": "server must be a non-empty string when provided",
            "server": server,
        }
    normalized_username = str(username).strip() if username is not None else None
    if username is not None and not normalized_username:
        return {
            "status": "error",
            "message": "username must be a non-empty string when provided",
            "username": username,
        }
    normalized_password = str(password).strip() if password is not None else None
    if password is not None and not normalized_password:
        return {
            "status": "error",
            "message": "password must be a non-empty string when provided",
            "password": "***",
        }
    if port is not None and (not isinstance(port, int) or port <= 0):
        return {
            "status": "error",
            "message": "port must be a positive integer when provided",
            "port": port,
        }
    if not isinstance(use_ssl, bool):
        return {
            "status": "error",
            "message": "use_ssl must be a boolean",
            "use_ssl": use_ssl,
        }
    if not isinstance(timeout, int) or timeout < 1:
        return {
            "status": "error",
            "message": "timeout must be an integer >= 1",
            "timeout": timeout,
        }

    result = await _API["email_list_folders"](
        server=normalized_server,
        port=port,
        username=normalized_username,
        password=normalized_password,
        use_ssl=use_ssl,
        timeout=timeout,
    )
    payload = dict(result or {})
    if "error" in payload and payload.get("error"):
        payload.setdefault("status", "error")
    else:
        payload.setdefault("status", "success")
    payload.setdefault("folders", [])
    payload.setdefault("folder_count", len(payload.get("folders") or []))
    payload.setdefault("use_ssl", use_ssl)
    payload.setdefault("timeout", timeout)
    if normalized_server is not None:
        payload.setdefault("server", normalized_server)
    return payload


async def email_analyze_export(**kwargs: Any) -> Dict[str, Any]:
    """Analyze exported email data."""
    file_path = kwargs.get("file_path")
    normalized_file_path = str(file_path or "").strip()
    if not normalized_file_path:
        return {
            "status": "error",
            "message": "file_path is required",
            "file_path": file_path,
        }
    result = await _API["email_analyze_export"](file_path=normalized_file_path)
    payload = dict(result or {})
    if "error" in payload and payload.get("error"):
        payload.setdefault("status", "error")
    else:
        payload.setdefault("status", "success")
    payload.setdefault("file_path", normalized_file_path)
    payload.setdefault("analysis", {})
    return payload


async def email_search_export(**kwargs: Any) -> Dict[str, Any]:
    """Search within exported email datasets."""
    file_path = kwargs.get("file_path")
    query = kwargs.get("query")
    field = kwargs.get("field", "all")

    normalized_file_path = str(file_path or "").strip()
    if not normalized_file_path:
        return {
            "status": "error",
            "message": "file_path is required",
            "file_path": file_path,
        }
    normalized_query = str(query or "").strip()
    if not normalized_query:
        return {
            "status": "error",
            "message": "query is required",
            "query": query,
        }
    normalized_field = str(field or "").strip().lower()
    valid_fields = {"all", "subject", "from", "to", "body"}
    if normalized_field not in valid_fields:
        return {
            "status": "error",
            "message": "field must be one of: all, subject, from, to, body",
            "field": field,
        }

    result = await _API["email_search_export"](
        file_path=normalized_file_path,
        query=normalized_query,
        field=normalized_field,
    )
    payload = dict(result or {})
    if "error" in payload and payload.get("error"):
        payload.setdefault("status", "error")
    else:
        payload.setdefault("status", "success")
    payload.setdefault("file_path", normalized_file_path)
    payload.setdefault("query", normalized_query)
    payload.setdefault("field", normalized_field)
    payload.setdefault("results", [])
    payload.setdefault("match_count", len(payload.get("results") or []))
    return payload


async def email_parse_eml(
    file_path: str,
    include_attachments: bool = True,
) -> Dict[str, Any]:
    """Parse an EML file and return normalized parsing envelope."""
    normalized_file_path = str(file_path or "").strip()
    if not normalized_file_path:
        return {
            "status": "error",
            "message": "file_path is required",
            "file_path": file_path,
        }
    if not isinstance(include_attachments, bool):
        return {
            "status": "error",
            "message": "include_attachments must be a boolean",
            "include_attachments": include_attachments,
        }

    result = await _API["email_parse_eml"](
        file_path=normalized_file_path,
        include_attachments=include_attachments,
    )
    payload = dict(result or {})
    if "error" in payload and payload.get("error"):
        payload.setdefault("status", "error")
    else:
        payload.setdefault("status", "success")
    payload.setdefault("file_path", normalized_file_path)
    payload.setdefault("include_attachments", include_attachments)
    payload.setdefault("email", {})
    return payload


def register_native_email_tools(manager: Any) -> None:
    """Register native email tools in unified hierarchical manager."""
    manager.register_tool(
        category="email_tools",
        name="email_test_connection",
        func=email_test_connection,
        description="Test connection to an email server.",
        input_schema={
            "type": "object",
            "properties": {
                "protocol": {"type": "string"},
                "server": {"type": ["string", "null"]},
                "port": {"type": ["integer", "null"]},
                "username": {"type": ["string", "null"]},
                "password": {"type": ["string", "null"]},
                "use_ssl": {"type": "boolean", "default": True},
                "timeout": {"type": "integer", "minimum": 1, "default": 30},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "email"],
    )

    manager.register_tool(
        category="email_tools",
        name="email_list_folders",
        func=email_list_folders,
        description="List mailbox folders from IMAP server.",
        input_schema={
            "type": "object",
            "properties": {
                "server": {"type": ["string", "null"]},
                "port": {"type": ["integer", "null"]},
                "username": {"type": ["string", "null"]},
                "password": {"type": ["string", "null"]},
                "use_ssl": {"type": "boolean", "default": True},
                "timeout": {"type": "integer", "minimum": 1, "default": 30},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "email"],
    )

    manager.register_tool(
        category="email_tools",
        name="email_analyze_export",
        func=email_analyze_export,
        description="Analyze exported email datasets.",
        input_schema={
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
            },
            "required": ["file_path"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "email"],
    )

    manager.register_tool(
        category="email_tools",
        name="email_search_export",
        func=email_search_export,
        description="Search exported email datasets.",
        input_schema={
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "query": {"type": "string"},
                "field": {
                    "type": "string",
                    "enum": ["all", "subject", "from", "to", "body"],
                    "default": "all",
                },
            },
            "required": ["file_path", "query"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "email"],
    )

    manager.register_tool(
        category="email_tools",
        name="email_parse_eml",
        func=email_parse_eml,
        description="Parse a single EML file and return extracted fields.",
        input_schema={
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "minLength": 1},
                "include_attachments": {"type": "boolean", "default": True},
            },
            "required": ["file_path"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "email"],
    )
