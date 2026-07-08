"""Native lizardperson-argparse-programs category implementations for unified mcp_server."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def _load_lizardperson_argparse_api() -> Dict[str, Any]:
    """Resolve source lizardperson argparse APIs with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.processors.legal_scrapers.bluebook_citation_validator.cli import (
            # type: ignore
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
    return {
        "status": "error",
        "success": False,
        "error": message,
        "entrypoint": "municipal_bluebook_citation_validator.main",
    }


def _normalize_payload(payload: Any) -> Dict[str, Any]:
    """Normalize delegate payloads to deterministic dict envelopes."""
    if isinstance(payload, dict):
        envelope = dict(payload)
        failed = bool(envelope.get("error")) or envelope.get("success") is False
        if failed:
            envelope["status"] = "error"
        elif "status" not in envelope:
            envelope["status"] = "success"
        return envelope
    if payload is None:
        return {"status": "success"}
    return {"status": "success", "result": payload}


async def municipal_bluebook_validator_info() -> Dict[str, Any]:
    """Return metadata about the municipal Bluebook citation validator CLI entrypoint."""
    try:
        envelope = _normalize_payload({
            "status": "success",
            "entrypoint": "municipal_bluebook_citation_validator.main",
            "callable": callable(_API.get("validator_main")),
            "fallback": not bool(_API),
        })
        envelope.setdefault("success", True)
        envelope.setdefault("entrypoint", "municipal_bluebook_citation_validator.main")
        envelope.setdefault("callable", False)
        envelope.setdefault("fallback", not bool(_API))
        return envelope
    except Exception as exc:
        logger.error("municipal_bluebook_validator_info failed: %s", exc)
        return _error_result(str(exc))


async def municipal_bluebook_validator_invoke(
    argv: List[str] | None = None,
    allow_execution: bool = False,
) -> Dict[str, Any]:
    """Safely invoke municipal Bluebook validator CLI entrypoint with guarded execution.

    Defaults to dry-run mode (`allow_execution=False`) to avoid side effects.
    """
    try:
        normalized_argv: List[str] = []
        if argv is None:
            normalized_argv = []
        elif not isinstance(argv, list):
            return _error_result("argv must be an array of strings")
        elif not all(isinstance(item, str) and item.strip() for item in argv):
            return _error_result("argv must contain only non-empty strings")
        else:
            normalized_argv = [item.strip() for item in argv]

        entrypoint = _API.get("validator_main")
        if not callable(entrypoint):
            return _error_result("validator entrypoint unavailable")

        if not bool(allow_execution):
            envelope = _normalize_payload({
                "status": "success",
                "entrypoint": "municipal_bluebook_citation_validator.main",
                "invoked": False,
                "dry_run": True,
                "argv": normalized_argv,
            })
            envelope.setdefault("success", True)
            envelope.setdefault("entrypoint", "municipal_bluebook_citation_validator.main")
            envelope.setdefault("invoked", False)
            envelope.setdefault("dry_run", True)
            envelope.setdefault("argv", normalized_argv)
            return envelope

        invocation_result = entrypoint(normalized_argv)
        if isinstance(invocation_result, dict):
            envelope = _normalize_payload(invocation_result)
        else:
            envelope = _normalize_payload(
                {
                    "status": "success",
                    "entrypoint": "municipal_bluebook_citation_validator.main",
                    "invoked": True,
                    "dry_run": False,
                    "argv": normalized_argv,
                    "exit_code": int(invocation_result) if isinstance(invocation_result, int) else 0,
                }
            )
        envelope.setdefault("success", True)
        envelope.setdefault("entrypoint", "municipal_bluebook_citation_validator.main")
        envelope.setdefault("invoked", True)
        envelope.setdefault("dry_run", False)
        envelope.setdefault("argv", normalized_argv)
        envelope.setdefault("exit_code", 0)
        return envelope
    except Exception as exc:
        logger.error("municipal_bluebook_validator_invoke failed: %s", exc)
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

    manager.register_tool(
        category="lizardperson_argparse_programs",
        name="municipal_bluebook_validator_invoke",
        func=municipal_bluebook_validator_invoke,
        description="Safely invoke municipal Bluebook citation validator with dry-run default.",
        input_schema={
            "type": "object",
            "properties": {
                "argv": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "default": [],
                },
                "allow_execution": {"type": "boolean", "default": False},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "lizardperson-argparse-programs"],
    )
