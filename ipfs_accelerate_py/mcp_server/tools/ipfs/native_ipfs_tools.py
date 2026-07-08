"""Native IPFS tool implementations for unified mcp_server."""

from __future__ import annotations

from typing import Any, Dict

from ipfs_accelerate_py.kit.ipfs_files_kit import get_ipfs_files_kit


def _error_result(message: str) -> Dict[str, Any]:
    return {"status": "error", "success": False, "data": None, "error": message}


def _normalize_kit_result(result: Any) -> Dict[str, Any]:
    """Normalize kit result objects to deterministic envelopes."""
    if isinstance(result, dict):
        normalized = dict(result)
        status = str(normalized.get("status", "")).strip().lower()
        success = normalized.get("success")
        error = normalized.get("error")
        has_error = error not in (None, "")

        if success is False or (status == "success" and has_error):
            normalized["status"] = "error"
            normalized["success"] = False
            normalized.setdefault("data", None)
            return normalized

        if success is True and normalized.get("data") is None:
            normalized["data"] = {}
        if success is True and not normalized.get("status"):
            normalized["status"] = "success"
        return normalized

    success = bool(getattr(result, "success", False))
    data = getattr(result, "data", None)
    error = getattr(result, "error", None)
    envelope = {
        "status": "success" if success and not error else "error",
        "success": success,
        "data": {} if success and data is None else data,
        "error": error,
    }
    return envelope


def ipfs_files_validate_cid(cid: str) -> Dict[str, Any]:
    """Validate CID format using the IPFS files kit.

    This is a native unified-tool implementation (not legacy registration capture)
    used in the Wave A migration path.
    """
    if not isinstance(cid, str) or not cid.strip():
        return _error_result("cid must be a non-empty string")

    kit = get_ipfs_files_kit()
    try:
        result = kit.validate_cid(cid=cid.strip())
        return _normalize_kit_result(result)
    except Exception as exc:
        return _error_result(str(exc))


def ipfs_files_list_files(path: str = "/") -> Dict[str, Any]:
    """List files for an IPFS MFS path using the IPFS files kit.

    This uses the same native unified-tool migration path as CID validation,
    while preserving the kit result envelope for callers.
    """
    if not isinstance(path, str) or not path.strip():
        return _error_result("path must be a non-empty string")

    kit = get_ipfs_files_kit()
    try:
        result = kit.list_files(path=path)
        return _normalize_kit_result(result)
    except Exception as exc:
        return _error_result(str(exc))


def ipfs_files_add_file(path: str, pin: bool = True) -> Dict[str, Any]:
    """Add a local file to IPFS using the IPFS files kit.

    This provides a native unified write/path operation in the Wave A migration
    path while preserving the kit result envelope for callers.
    """
    if not isinstance(path, str) or not path.strip():
        return _error_result("path must be a non-empty string")
    if not isinstance(pin, bool):
        return _error_result("pin must be a boolean")

    kit = get_ipfs_files_kit()
    try:
        result = kit.add_file(path=path, pin=pin)
        return _normalize_kit_result(result)
    except Exception as exc:
        return _error_result(str(exc))


def ipfs_files_pin_file(cid: str) -> Dict[str, Any]:
    """Pin a CID in IPFS using the IPFS files kit.

    This extends the native unified Wave A tool set with a pinning operation
    while preserving the kit result envelope for callers.
    """
    if not isinstance(cid, str) or not cid.strip():
        return _error_result("cid must be a non-empty string")

    kit = get_ipfs_files_kit()
    try:
        result = kit.pin_file(cid=cid.strip())
        return _normalize_kit_result(result)
    except Exception as exc:
        return _error_result(str(exc))


def ipfs_files_unpin_file(cid: str) -> Dict[str, Any]:
    """Unpin a CID in IPFS using the IPFS files kit.

    This extends the native unified Wave A tool set with an unpin operation
    while preserving the kit result envelope for callers.
    """
    if not isinstance(cid, str) or not cid.strip():
        return _error_result("cid must be a non-empty string")

    kit = get_ipfs_files_kit()
    try:
        result = kit.unpin_file(cid=cid.strip())
        return _normalize_kit_result(result)
    except Exception as exc:
        return _error_result(str(exc))


def ipfs_files_get_file(cid: str, output_path: str) -> Dict[str, Any]:
    """Get a file from IPFS by CID using the IPFS files kit.

    This extends the native unified Wave A tool set with a read/get operation
    while preserving the kit result envelope for callers.
    """
    if not isinstance(cid, str) or not cid.strip():
        return _error_result("cid must be a non-empty string")
    if not isinstance(output_path, str) or not output_path.strip():
        return _error_result("output_path must be a non-empty string")

    kit = get_ipfs_files_kit()
    try:
        result = kit.get_file(cid=cid.strip(), output_path=output_path)
        return _normalize_kit_result(result)
    except Exception as exc:
        return _error_result(str(exc))


def ipfs_files_cat(cid: str) -> Dict[str, Any]:
    """Read file content from IPFS by CID using the IPFS files kit.

    This extends the native unified Wave A tool set with direct content
    retrieval (`ipfs cat` equivalent) while preserving the kit result envelope.
    """
    if not isinstance(cid, str) or not cid.strip():
        return _error_result("cid must be a non-empty string")

    kit = get_ipfs_files_kit()
    try:
        result = kit.cat_file(cid=cid.strip())
        return _normalize_kit_result(result)
    except Exception as exc:
        return _error_result(str(exc))


def register_native_ipfs_tools(manager: Any) -> None:
    """Register native IPFS tools in the unified hierarchical manager."""
    manager.register_tool(
        category="ipfs",
        name="ipfs_files_validate_cid",
        func=ipfs_files_validate_cid,
        description="Validate CID format using unified native IPFS tool implementation.",
        input_schema={
            "type": "object",
            "properties": {
                "cid": {"type": "string", "minLength": 1},
            },
            "required": ["cid"],
        },
        runtime="fastapi",
        tags=["native", "wave-a", "ipfs"],
    )

    manager.register_tool(
        category="ipfs",
        name="ipfs_files_list_files",
        func=ipfs_files_list_files,
        description="List files in IPFS MFS path using unified native IPFS tool implementation.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "default": "/", "minLength": 1},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "wave-a", "ipfs"],
    )

    manager.register_tool(
        category="ipfs",
        name="ipfs_files_add_file",
        func=ipfs_files_add_file,
        description="Add local file to IPFS using unified native IPFS tool implementation.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "minLength": 1},
                "pin": {"type": "boolean", "default": True},
            },
            "required": ["path"],
        },
        runtime="fastapi",
        tags=["native", "wave-a", "ipfs"],
    )

    manager.register_tool(
        category="ipfs",
        name="ipfs_files_pin_file",
        func=ipfs_files_pin_file,
        description="Pin CID in IPFS using unified native IPFS tool implementation.",
        input_schema={
            "type": "object",
            "properties": {
                "cid": {"type": "string", "minLength": 1},
            },
            "required": ["cid"],
        },
        runtime="fastapi",
        tags=["native", "wave-a", "ipfs"],
    )

    manager.register_tool(
        category="ipfs",
        name="ipfs_files_unpin_file",
        func=ipfs_files_unpin_file,
        description="Unpin CID in IPFS using unified native IPFS tool implementation.",
        input_schema={
            "type": "object",
            "properties": {
                "cid": {"type": "string", "minLength": 1},
            },
            "required": ["cid"],
        },
        runtime="fastapi",
        tags=["native", "wave-a", "ipfs"],
    )

    manager.register_tool(
        category="ipfs",
        name="ipfs_files_get_file",
        func=ipfs_files_get_file,
        description="Get file from IPFS by CID using unified native IPFS tool implementation.",
        input_schema={
            "type": "object",
            "properties": {
                "cid": {"type": "string", "minLength": 1},
                "output_path": {"type": "string", "minLength": 1},
            },
            "required": ["cid", "output_path"],
        },
        runtime="fastapi",
        tags=["native", "wave-a", "ipfs"],
    )

    manager.register_tool(
        category="ipfs",
        name="ipfs_files_cat",
        func=ipfs_files_cat,
        description="Read file content from IPFS by CID using unified native IPFS tool implementation.",
        input_schema={
            "type": "object",
            "properties": {
                "cid": {"type": "string", "minLength": 1},
            },
            "required": ["cid"],
        },
        runtime="fastapi",
        tags=["native", "wave-a", "ipfs"],
    )
