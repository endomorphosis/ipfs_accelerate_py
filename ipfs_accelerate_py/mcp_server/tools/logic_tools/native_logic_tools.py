"""Native logic-tools category implementations for unified mcp_server."""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def _load_logic_tools_api() -> Dict[str, Any]:
    """Resolve source logic-tools APIs with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.logic_tools.logic_capabilities_tool import (  # type: ignore
            logic_capabilities as _logic_capabilities,
            logic_health as _logic_health,
        )
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.logic_tools.tdfol_parse_tool import (  # type: ignore
            tdfol_parse as _tdfol_parse,
        )
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.logic_tools.tdfol_convert_tool import (  # type: ignore
            tdfol_convert as _tdfol_convert,
        )
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.logic_tools.tdfol_prove_tool import (  # type: ignore
            tdfol_prove as _tdfol_prove,
        )

        return {
            "logic_capabilities": _logic_capabilities,
            "logic_health": _logic_health,
            "tdfol_parse": _tdfol_parse,
            "tdfol_convert": _tdfol_convert,
            "tdfol_prove": _tdfol_prove,
        }
    except Exception:
        logger.warning("Source logic_tools import unavailable, using fallback logic functions")

        async def _logic_capabilities_fallback() -> Dict[str, Any]:
            return {
                "status": "success",
                "logics": {},
                "conversions": [],
                "fallback": True,
            }

        async def _logic_health_fallback() -> Dict[str, Any]:
            return {
                "status": "success",
                "healthy": 0,
                "total": 0,
                "fallback": True,
            }

        async def _tdfol_parse_fallback(
            text: str,
            format: str = "symbolic",
            language: str = "en",
        ) -> Dict[str, Any]:
            if not str(text or "").strip():
                return {
                    "success": False,
                    "error": "'text' is required.",
                }
            return {
                "success": False,
                "error": "tdfol_parse: LogicProcessor not available.",
                "format": format,
                "language": language,
            }

        async def _tdfol_convert_fallback(
            formula: str,
            source_format: str = "tdfol",
            target_format: str = "fol",
        ) -> Dict[str, Any]:
            if not str(formula or "").strip():
                return {
                    "success": False,
                    "error": "'formula' is required.",
                }
            return {
                "success": False,
                "error": "tdfol_convert: LogicProcessor not available.",
                "source_format": source_format,
                "target_format": target_format,
            }

        async def _tdfol_prove_fallback(
            formula: str,
            axioms: list[str] | None = None,
            strategy: str = "auto",
            timeout_ms: int = 5000,
            max_depth: int = 10,
            include_proof_steps: bool = True,
        ) -> Dict[str, Any]:
            if not str(formula or "").strip():
                return {
                    "success": False,
                    "error": "'formula' is required.",
                }
            _ = axioms, strategy, timeout_ms, max_depth, include_proof_steps
            return {
                "success": False,
                "error": "tdfol_prove: LogicProcessor not available.",
                "proved": False,
                "formula": formula,
            }

        return {
            "logic_capabilities": _logic_capabilities_fallback,
            "logic_health": _logic_health_fallback,
            "tdfol_parse": _tdfol_parse_fallback,
            "tdfol_convert": _tdfol_convert_fallback,
            "tdfol_prove": _tdfol_prove_fallback,
        }


_API = _load_logic_tools_api()


async def logic_capabilities() -> Dict[str, Any]:
    """Return discovered logic-module capabilities for the unified runtime."""
    result = _API["logic_capabilities"]()
    if hasattr(result, "__await__"):
        return await result
    return result


async def logic_health() -> Dict[str, Any]:
    """Return logic-module health status for the unified runtime."""
    result = _API["logic_health"]()
    if hasattr(result, "__await__"):
        return await result
    return result


async def tdfol_parse(
    text: str,
    format: str = "symbolic",
    language: str = "en",
) -> Dict[str, Any]:
    """Parse symbolic or natural-language input into TDFOL notation."""
    normalized_text = str(text or "").strip()
    if not normalized_text:
        return {
            "success": False,
            "error": "'text' is required.",
        }

    result = _API["tdfol_parse"](
        text=normalized_text,
        format=str(format or "symbolic"),
        language=str(language or "en"),
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def tdfol_convert(
    formula: str,
    source_format: str = "tdfol",
    target_format: str = "fol",
) -> Dict[str, Any]:
    """Convert a formula across supported logic formats."""
    normalized_formula = str(formula or "").strip()
    if not normalized_formula:
        return {
            "success": False,
            "error": "'formula' is required.",
        }

    result = _API["tdfol_convert"](
        formula=normalized_formula,
        source_format=str(source_format or "tdfol"),
        target_format=str(target_format or "fol"),
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def tdfol_prove(
    formula: str,
    axioms: list[str] | None = None,
    strategy: str = "auto",
    timeout_ms: int = 5000,
    max_depth: int = 10,
    include_proof_steps: bool = True,
) -> Dict[str, Any]:
    """Prove a TDFOL formula with deterministic fallback envelopes."""
    normalized_formula = str(formula or "").strip()
    if not normalized_formula:
        return {
            "success": False,
            "error": "'formula' is required.",
        }

    result = _API["tdfol_prove"](
        formula=normalized_formula,
        axioms=axioms,
        strategy=str(strategy or "auto"),
        timeout_ms=int(timeout_ms),
        max_depth=int(max_depth),
        include_proof_steps=bool(include_proof_steps),
    )
    if hasattr(result, "__await__"):
        result = await result

    if isinstance(result, dict) and result.get("success") is False and "formula" not in result:
        result = dict(result)
        result["formula"] = normalized_formula

    return result


def register_native_logic_tools(manager: Any) -> None:
    """Register native logic-tools category tools in unified manager."""
    manager.register_tool(
        category="logic_tools",
        name="logic_capabilities",
        func=logic_capabilities,
        description="List capabilities for available logic modules.",
        input_schema={"type": "object", "properties": {}, "required": []},
        runtime="fastapi",
        tags=["native", "mcpp", "logic-tools"],
    )

    manager.register_tool(
        category="logic_tools",
        name="logic_health",
        func=logic_health,
        description="Get health status for available logic modules.",
        input_schema={"type": "object", "properties": {}, "required": []},
        runtime="fastapi",
        tags=["native", "mcpp", "logic-tools"],
    )

    manager.register_tool(
        category="logic_tools",
        name="tdfol_parse",
        func=tdfol_parse,
        description="Parse input text into TDFOL notation.",
        input_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "format": {"type": "string", "default": "symbolic"},
                "language": {"type": "string", "default": "en"},
            },
            "required": ["text"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "logic-tools"],
    )

    manager.register_tool(
        category="logic_tools",
        name="tdfol_convert",
        func=tdfol_convert,
        description="Convert formulas between supported logic notations.",
        input_schema={
            "type": "object",
            "properties": {
                "formula": {"type": "string"},
                "source_format": {"type": "string", "default": "tdfol"},
                "target_format": {"type": "string", "default": "fol"},
            },
            "required": ["formula"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "logic-tools"],
    )

    manager.register_tool(
        category="logic_tools",
        name="tdfol_prove",
        func=tdfol_prove,
        description="Attempt to prove a TDFOL formula with optional axioms.",
        input_schema={
            "type": "object",
            "properties": {
                "formula": {"type": "string"},
                "axioms": {"type": ["array", "null"], "items": {"type": "string"}},
                "strategy": {"type": "string", "default": "auto"},
                "timeout_ms": {"type": "integer", "default": 5000},
                "max_depth": {"type": "integer", "default": 10},
                "include_proof_steps": {"type": "boolean", "default": True},
            },
            "required": ["formula"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "logic-tools"],
    )
