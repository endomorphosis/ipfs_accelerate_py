"""Native lizardperson-argparse-programs category implementations for unified mcp_server."""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def _load_lizardperson_argparse_api() -> Dict[str, Any]:
    """Resolve source lizardperson argparse APIs with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.processors.legal_scrapers.bluebook_citation_validator.cli import (  # type: ignore
            main as _validator_main,
        )

        return {"validator_main": _validator_main}
    except Exception:
        logger.warning(
            "Source lizardperson_argparse_programs import unavailable, using fallback argparse metadata"
        )
        return {}


_API = _load_lizardperson_argparse_api()


def _error_result(message: str) -> Dict[str, Any]:
    return {"status": "error", "error": message, "entrypoint": "municipal_bluebook_citation_validator.main"}


async def municipal_bluebook_validator_info() -> Dict[str, Any]:
    """Return metadata about the municipal Bluebook citation validator CLI entrypoint."""
    try:
        return {
            "status": "success",
            "entrypoint": "municipal_bluebook_citation_validator.main",
            "callable": callable(_API.get("validator_main")),
            "fallback": not bool(_API),
        }
    except Exception as exc:
        logger.error("municipal_bluebook_validator_info failed: %s", exc)
        return _error_result(str(exc))


def register_native_lizardperson_argparse_programs(manager: Any) -> None:
    """Register native lizardperson-argparse-programs tools in unified manager."""
    manager.register_tool(
        category="lizardperson_argparse_programs",
        name="municipal_bluebook_validator_info",
        func=municipal_bluebook_validator_info,
        description="Inspect metadata for municipal Bluebook citation validator argparse program.",
        input_schema={"type": "object", "properties": {}, "required": []},
        runtime="fastapi",
        tags=["native", "mcpp", "lizardperson-argparse-programs"],
    )
