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

        return {
            "email_test_connection": _email_test_connection,
            "email_list_folders": _email_list_folders,
            "email_analyze_export": _email_analyze_export,
            "email_search_export": _email_search_export,
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

        return {
            "email_test_connection": _test_conn_fallback,
            "email_list_folders": _list_folders_fallback,
            "email_analyze_export": _analyze_export_fallback,
            "email_search_export": _search_export_fallback,
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
    return await _API["email_test_connection"](
        protocol=protocol,
        server=server,
        port=port,
        username=username,
        password=password,
        use_ssl=use_ssl,
        timeout=timeout,
    )


async def email_list_folders(
    server: Optional[str] = None,
    port: Optional[int] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    use_ssl: bool = True,
    timeout: int = 30,
) -> Dict[str, Any]:
    """List available mailbox folders from IMAP server."""
    return await _API["email_list_folders"](
        server=server,
        port=port,
        username=username,
        password=password,
        use_ssl=use_ssl,
        timeout=timeout,
    )


async def email_analyze_export(**kwargs: Any) -> Dict[str, Any]:
    """Analyze exported email data."""
    return await _API["email_analyze_export"](**kwargs)


async def email_search_export(**kwargs: Any) -> Dict[str, Any]:
    """Search within exported email datasets."""
    return await _API["email_search_export"](**kwargs)


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
                "use_ssl": {"type": "boolean"},
                "timeout": {"type": "integer"},
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
                "use_ssl": {"type": "boolean"},
                "timeout": {"type": "integer"},
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
        input_schema={"type": "object", "properties": {}, "required": []},
        runtime="fastapi",
        tags=["native", "mcpp", "email"],
    )

    manager.register_tool(
        category="email_tools",
        name="email_search_export",
        func=email_search_export,
        description="Search exported email datasets.",
        input_schema={"type": "object", "properties": {}, "required": []},
        runtime="fastapi",
        tags=["native", "mcpp", "email"],
    )
