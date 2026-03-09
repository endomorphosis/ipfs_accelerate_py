"""Native logic-tools category implementations for unified mcp_server."""

from __future__ import annotations

import logging
from importlib import import_module
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_LOCAL_TDFOL_KB_STATE: Dict[str, List[str]] = {
    "axioms": [],
    "theorems": [],
}


def _load_logic_tools_api() -> Dict[str, Any]:
    """Resolve source logic-tools APIs with compatibility fallback."""
    try:
        logic_capabilities_module = import_module(
            "ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.logic_tools.logic_capabilities_tool"
        )
        _logic_capabilities = getattr(logic_capabilities_module, "logic_capabilities")
        _logic_health = getattr(logic_capabilities_module, "logic_health")
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.logic_tools.tdfol_parse_tool import (  # type: ignore
            tdfol_parse as _tdfol_parse,
        )
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.logic_tools.tdfol_convert_tool import (  # type: ignore
            tdfol_convert as _tdfol_convert,
        )
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.logic_tools.tdfol_prove_tool import (  # type: ignore
            tdfol_prove as _tdfol_prove,
        )
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.logic_tools.tdfol_kb_tool import (  # type: ignore
            tdfol_kb_add_axiom as _tdfol_kb_add_axiom,
            tdfol_kb_add_theorem as _tdfol_kb_add_theorem,
            tdfol_kb_export as _tdfol_kb_export,
            tdfol_kb_query as _tdfol_kb_query,
        )
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.logic_tools.cec_prove_tool import (  # type: ignore
            cec_check_theorem as _cec_check_theorem,
            cec_prove as _cec_prove,
        )
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.logic_tools.cec_parse_tool import (  # type: ignore
            cec_parse as _cec_parse,
            cec_validate_formula as _cec_validate_formula,
        )
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.logic_tools.cec_analysis_tool import (  # type: ignore
            cec_analyze_formula as _cec_analyze_formula,
            cec_formula_complexity as _cec_formula_complexity,
        )

        return {
            "logic_capabilities": _logic_capabilities,
            "logic_health": _logic_health,
            "tdfol_parse": _tdfol_parse,
            "tdfol_convert": _tdfol_convert,
            "tdfol_prove": _tdfol_prove,
            "tdfol_kb_add_axiom": _tdfol_kb_add_axiom,
            "tdfol_kb_add_theorem": _tdfol_kb_add_theorem,
            "tdfol_kb_query": _tdfol_kb_query,
            "tdfol_kb_export": _tdfol_kb_export,
            "cec_prove": _cec_prove,
            "cec_check_theorem": _cec_check_theorem,
            "cec_parse": _cec_parse,
            "cec_validate_formula": _cec_validate_formula,
            "cec_analyze_formula": _cec_analyze_formula,
            "cec_formula_complexity": _cec_formula_complexity,
        }
    except Exception:
        logger.warning("Source logic_tools import unavailable, using fallback logic functions")

        kb_state: Dict[str, List[str]] = {
            "axioms": [],
            "theorems": [],
        }

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

        async def _tdfol_kb_add_axiom_fallback(formula: str) -> Dict[str, Any]:
            if not str(formula or "").strip():
                return {"success": False, "error": "'formula' is required."}
            normalized_formula = str(formula).strip()
            kb_state["axioms"].append(normalized_formula)
            return {
                "success": True,
                "formula": normalized_formula,
                "axiom_count": len(kb_state["axioms"]),
            }

        async def _tdfol_kb_add_theorem_fallback(formula: str) -> Dict[str, Any]:
            if not str(formula or "").strip():
                return {"success": False, "error": "'formula' is required."}
            normalized_formula = str(formula).strip()
            kb_state["theorems"].append(normalized_formula)
            return {
                "success": True,
                "formula": normalized_formula,
                "theorem_count": len(kb_state["theorems"]),
            }

        async def _tdfol_kb_query_fallback() -> Dict[str, Any]:
            return {
                "success": True,
                "stats": {
                    "axiom_count": len(kb_state["axioms"]),
                    "theorem_count": len(kb_state["theorems"]),
                    "total_count": len(kb_state["axioms"]) + len(kb_state["theorems"]),
                },
                "axioms": list(kb_state["axioms"]),
                "theorems": list(kb_state["theorems"]),
            }

        async def _tdfol_kb_export_fallback(export_format: str = "json") -> Dict[str, Any]:
            return {
                "success": True,
                "format": export_format,
                "data": {
                    "axioms": list(kb_state["axioms"]),
                    "theorems": list(kb_state["theorems"]),
                },
            }

        async def _cec_prove_fallback(
            goal: str,
            axioms: Optional[List[str]] = None,
            strategy: str = "auto",
            timeout: int = 30,
        ) -> Dict[str, Any]:
            if not str(goal or "").strip():
                return {"success": False, "error": "'goal' is required.", "proved": False}
            _ = axioms, strategy, timeout
            return {
                "success": False,
                "error": "cec_prove: LogicProcessor not available.",
                "proved": False,
                "goal": goal,
            }

        async def _cec_check_theorem_fallback(
            formula: str,
            axioms: Optional[List[str]] = None,
        ) -> Dict[str, Any]:
            if not str(formula or "").strip():
                return {"success": False, "error": "'formula' is required.", "is_theorem": False}
            _ = axioms
            return {
                "success": False,
                "error": "cec_check_theorem: LogicProcessor not available.",
                "is_theorem": False,
                "formula": formula,
            }

        async def _cec_parse_fallback(
            text: str,
            language: str = "en",
            domain: str = "general",
        ) -> Dict[str, Any]:
            if not str(text or "").strip():
                return {"success": False, "error": "'text' is required.", "formula": None}
            return {
                "success": False,
                "error": "cec_parse: LogicProcessor not available.",
                "formula": None,
                "language_used": language,
                "domain": domain,
            }

        async def _cec_validate_formula_fallback(formula: str) -> Dict[str, Any]:
            if not str(formula or "").strip():
                return {"success": False, "valid": False, "errors": ["'formula' is required."], "warnings": []}
            return {
                "success": False,
                "valid": False,
                "errors": ["cec_validate_formula: LogicProcessor not available."],
                "warnings": [],
                "formula": formula,
            }

        async def _cec_analyze_formula_fallback(formula: str) -> Dict[str, Any]:
            if not str(formula or "").strip():
                return {"success": False, "error": "'formula' is required."}
            operators = [op for op in ["->", "&", "|", "¬"] if op in formula]
            depth = formula.count("(")
            return {
                "success": True,
                "formula": formula,
                "depth": depth,
                "size": max(1, len(formula.split())),
                "operators": operators,
                "parsed_ok": True,
            }

        async def _cec_formula_complexity_fallback(formula: str) -> Dict[str, Any]:
            if not str(formula or "").strip():
                return {"success": False, "error": "'formula' is required."}
            complexity = "low" if len(formula) < 20 else "medium" if len(formula) < 60 else "high"
            return {
                "success": True,
                "formula": formula,
                "complexity": complexity,
                "depth": formula.count("("),
                "size": max(1, len(formula.split())),
            }

        return {
            "logic_capabilities": _logic_capabilities_fallback,
            "logic_health": _logic_health_fallback,
            "tdfol_parse": _tdfol_parse_fallback,
            "tdfol_convert": _tdfol_convert_fallback,
            "tdfol_prove": _tdfol_prove_fallback,
            "tdfol_kb_add_axiom": _tdfol_kb_add_axiom_fallback,
            "tdfol_kb_add_theorem": _tdfol_kb_add_theorem_fallback,
            "tdfol_kb_query": _tdfol_kb_query_fallback,
            "tdfol_kb_export": _tdfol_kb_export_fallback,
            "cec_prove": _cec_prove_fallback,
            "cec_check_theorem": _cec_check_theorem_fallback,
            "cec_parse": _cec_parse_fallback,
            "cec_validate_formula": _cec_validate_formula_fallback,
            "cec_analyze_formula": _cec_analyze_formula_fallback,
            "cec_formula_complexity": _cec_formula_complexity_fallback,
        }


_API = _load_logic_tools_api()


def _error_result(message: str, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Return deterministic logic-tool error envelopes."""
    payload: Dict[str, Any] = {"status": "error", "success": False, "error": message}
    if context:
        payload.update(context)
    return payload


async def _await_maybe(result: Any) -> Any:
    """Await coroutine-like values while preserving sync fallback behavior."""
    if hasattr(result, "__await__"):
        return await result
    return result


def _normalize_result(payload: Any) -> Dict[str, Any]:
    """Normalize delegate payloads to deterministic dictionary envelopes."""
    normalized: Dict[str, Any] = dict(payload or {}) if isinstance(payload, dict) else {"result": payload}
    failed = normalized.get("success") is False or bool(normalized.get("error")) or bool(normalized.get("errors"))
    if failed:
        normalized["status"] = "error"
        normalized["success"] = False
    else:
        normalized.setdefault("status", "success")
        normalized.setdefault("success", True)
    return normalized


def _logic_delegate_unavailable(payload: Dict[str, Any]) -> bool:
    error = str(payload.get("error", ""))
    return payload.get("success") is False and "LogicProcessor not available" in error


def _local_kb_add(bucket: str, formula: str) -> Dict[str, Any]:
    _LOCAL_TDFOL_KB_STATE[bucket].append(formula)
    return {
        "success": True,
        "formula": formula,
        f"{bucket[:-1]}_count": len(_LOCAL_TDFOL_KB_STATE[bucket]),
    }


def _local_kb_query() -> Dict[str, Any]:
    return {
        "success": True,
        "stats": {
            "axiom_count": len(_LOCAL_TDFOL_KB_STATE["axioms"]),
            "theorem_count": len(_LOCAL_TDFOL_KB_STATE["theorems"]),
            "total_count": len(_LOCAL_TDFOL_KB_STATE["axioms"]) + len(_LOCAL_TDFOL_KB_STATE["theorems"]),
        },
        "axioms": list(_LOCAL_TDFOL_KB_STATE["axioms"]),
        "theorems": list(_LOCAL_TDFOL_KB_STATE["theorems"]),
    }


def _local_kb_export(export_format: str) -> Dict[str, Any]:
    return {
        "success": True,
        "format": export_format,
        "data": {
            "axioms": list(_LOCAL_TDFOL_KB_STATE["axioms"]),
            "theorems": list(_LOCAL_TDFOL_KB_STATE["theorems"]),
        },
    }


def _normalize_non_empty_string(value: Any, field_name: str) -> str | None:
    normalized = str(value or "").strip()
    if not normalized:
        return None
    return normalized


def _normalize_string_list(value: Any, field_name: str) -> List[str] | None:
    if value is None:
        return None
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        return []
    return [str(item).strip() for item in value]


async def logic_capabilities() -> Dict[str, Any]:
    """Return discovered logic-module capabilities for the unified runtime."""
    try:
        payload = await _await_maybe(_API["logic_capabilities"]())
    except Exception as exc:
        return _error_result(f"logic_capabilities failed: {exc}")
    return _normalize_result(payload)


async def logic_health() -> Dict[str, Any]:
    """Return logic-module health status for the unified runtime."""
    try:
        payload = await _await_maybe(_API["logic_health"]())
    except Exception as exc:
        return _error_result(f"logic_health failed: {exc}")
    return _normalize_result(payload)


async def tdfol_parse(
    text: str,
    format: str = "symbolic",
    language: str = "en",
) -> Dict[str, Any]:
    """Parse symbolic or natural-language input into TDFOL notation."""
    normalized_text = str(text or "").strip()
    if not normalized_text:
        return _error_result("'text' is required.")

    normalized_format = "symbolic" if format is None else str(format).strip()
    normalized_language = "en" if language is None else str(language).strip()

    if not normalized_format:
        return _error_result("'format' must be a non-empty string.")
    if not normalized_language:
        return _error_result("'language' must be a non-empty string.")

    try:
        payload = await _await_maybe(
            _API["tdfol_parse"](
                text=normalized_text,
                format=normalized_format,
                language=normalized_language,
            )
        )
    except Exception as exc:
        return _error_result(f"tdfol_parse failed: {exc}")

    return _normalize_result(payload)


async def tdfol_convert(
    formula: str,
    source_format: str = "tdfol",
    target_format: str = "fol",
) -> Dict[str, Any]:
    """Convert a formula across supported logic formats."""
    normalized_formula = str(formula or "").strip()
    if not normalized_formula:
        return _error_result("'formula' is required.")

    normalized_source = "tdfol" if source_format is None else str(source_format).strip()
    normalized_target = "fol" if target_format is None else str(target_format).strip()
    if not normalized_source:
        return _error_result("'source_format' must be a non-empty string.")
    if not normalized_target:
        return _error_result("'target_format' must be a non-empty string.")

    try:
        payload = await _await_maybe(
            _API["tdfol_convert"](
                formula=normalized_formula,
                source_format=normalized_source,
                target_format=normalized_target,
            )
        )
    except Exception as exc:
        return _error_result(f"tdfol_convert failed: {exc}")

    return _normalize_result(payload)


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
        return _error_result("'formula' is required.")
    if axioms is not None:
        if not isinstance(axioms, list) or any(not isinstance(a, str) or not a.strip() for a in axioms):
            return _error_result("'axioms' must be a list of non-empty strings when provided.")

    normalized_strategy = "auto" if strategy is None else str(strategy).strip()
    if not normalized_strategy:
        return _error_result("'strategy' must be a non-empty string.")
    if not isinstance(timeout_ms, int) or timeout_ms < 1:
        return _error_result("'timeout_ms' must be an integer greater than or equal to 1.")
    if not isinstance(max_depth, int) or max_depth < 1:
        return _error_result("'max_depth' must be an integer greater than or equal to 1.")
    if not isinstance(include_proof_steps, bool):
        return _error_result("'include_proof_steps' must be a boolean.")

    try:
        payload = await _await_maybe(
            _API["tdfol_prove"](
                formula=normalized_formula,
                axioms=axioms,
                strategy=normalized_strategy,
                timeout_ms=timeout_ms,
                max_depth=max_depth,
                include_proof_steps=include_proof_steps,
            )
        )
    except Exception as exc:
        return _error_result(f"tdfol_prove failed: {exc}", {"formula": normalized_formula})

    normalized = _normalize_result(payload)
    if normalized.get("success") is False and "formula" not in normalized:
        normalized["formula"] = normalized_formula
    return normalized


async def tdfol_kb_add_axiom(formula: str) -> Dict[str, Any]:
    """Add an axiom to the TDFOL knowledge base."""
    normalized_formula = _normalize_non_empty_string(formula, "formula")
    if normalized_formula is None:
        return _error_result("'formula' is required.")
    try:
        payload = await _await_maybe(_API["tdfol_kb_add_axiom"](formula=normalized_formula))
    except Exception as exc:
        return _error_result(f"tdfol_kb_add_axiom failed: {exc}", {"formula": normalized_formula})
    normalized = _normalize_result(payload)
    if _logic_delegate_unavailable(normalized):
        normalized = _local_kb_add("axioms", normalized_formula)
    normalized.setdefault("formula", normalized_formula)
    return normalized


async def tdfol_kb_add_theorem(formula: str) -> Dict[str, Any]:
    """Add a theorem to the TDFOL knowledge base."""
    normalized_formula = _normalize_non_empty_string(formula, "formula")
    if normalized_formula is None:
        return _error_result("'formula' is required.")
    try:
        payload = await _await_maybe(_API["tdfol_kb_add_theorem"](formula=normalized_formula))
    except Exception as exc:
        return _error_result(f"tdfol_kb_add_theorem failed: {exc}", {"formula": normalized_formula})
    normalized = _normalize_result(payload)
    if _logic_delegate_unavailable(normalized):
        normalized = _local_kb_add("theorems", normalized_formula)
    normalized.setdefault("formula", normalized_formula)
    return normalized


async def tdfol_kb_query() -> Dict[str, Any]:
    """Query TDFOL knowledge-base contents and statistics."""
    try:
        payload = await _await_maybe(_API["tdfol_kb_query"]())
    except Exception as exc:
        return _error_result(f"tdfol_kb_query failed: {exc}")
    normalized = _normalize_result(payload)
    if _logic_delegate_unavailable(normalized):
        normalized = _local_kb_query()
    normalized.setdefault("stats", {})
    return normalized


async def tdfol_kb_export(export_format: str = "json") -> Dict[str, Any]:
    """Export the TDFOL knowledge base."""
    normalized_format = "json" if export_format is None else str(export_format).strip().lower()
    if normalized_format not in {"json", "tptp", "smt2"}:
        return _error_result("'export_format' must be one of: json, tptp, smt2.")
    try:
        payload = await _await_maybe(_API["tdfol_kb_export"](export_format=normalized_format))
    except Exception as exc:
        return _error_result(f"tdfol_kb_export failed: {exc}", {"format": normalized_format})
    normalized = _normalize_result(payload)
    if _logic_delegate_unavailable(normalized):
        normalized = _local_kb_export(normalized_format)
    normalized.setdefault("format", normalized_format)
    return normalized


async def cec_prove(
    goal: str,
    axioms: Optional[List[str]] = None,
    strategy: str = "auto",
    timeout: int = 30,
) -> Dict[str, Any]:
    """Prove a DCEC/CEC theorem with deterministic validation."""
    normalized_goal = _normalize_non_empty_string(goal, "goal")
    if normalized_goal is None:
        return _error_result("'goal' is required.", {"proved": False})
    normalized_axioms = _normalize_string_list(axioms, "axioms")
    if axioms is not None and normalized_axioms == []:
        return _error_result("'axioms' must be a list of non-empty strings when provided.")
    normalized_strategy = "auto" if strategy is None else str(strategy).strip().lower()
    if not normalized_strategy:
        return _error_result("'strategy' must be a non-empty string.")
    if not isinstance(timeout, int) or timeout < 1:
        return _error_result("'timeout' must be an integer greater than or equal to 1.")
    try:
        payload = await _await_maybe(
            _API["cec_prove"](
                goal=normalized_goal,
                axioms=normalized_axioms,
                strategy=normalized_strategy,
                timeout=timeout,
            )
        )
    except Exception as exc:
        return _error_result(f"cec_prove failed: {exc}", {"goal": normalized_goal, "proved": False})
    normalized = _normalize_result(payload)
    normalized.setdefault("goal", normalized_goal)
    normalized.setdefault("proved", False)
    return normalized


async def cec_check_theorem(
    formula: str,
    axioms: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Check whether a CEC formula is a theorem."""
    normalized_formula = _normalize_non_empty_string(formula, "formula")
    if normalized_formula is None:
        return _error_result("'formula' is required.", {"is_theorem": False})
    normalized_axioms = _normalize_string_list(axioms, "axioms")
    if axioms is not None and normalized_axioms == []:
        return _error_result("'axioms' must be a list of non-empty strings when provided.")
    try:
        payload = await _await_maybe(_API["cec_check_theorem"](formula=normalized_formula, axioms=normalized_axioms))
    except Exception as exc:
        return _error_result(f"cec_check_theorem failed: {exc}", {"formula": normalized_formula})
    normalized = _normalize_result(payload)
    normalized.setdefault("formula", normalized_formula)
    normalized.setdefault("is_theorem", False)
    return normalized


async def cec_parse(
    text: str,
    language: str = "en",
    domain: str = "general",
) -> Dict[str, Any]:
    """Parse natural language text into a CEC/DCEC formula."""
    normalized_text = _normalize_non_empty_string(text, "text")
    if normalized_text is None:
        return _error_result("'text' is required.", {"formula": None})
    normalized_language = "en" if language is None else str(language).strip().lower()
    normalized_domain = "general" if domain is None else str(domain).strip().lower()
    if not normalized_language:
        return _error_result("'language' must be a non-empty string.")
    if not normalized_domain:
        return _error_result("'domain' must be a non-empty string.")
    try:
        payload = await _await_maybe(
            _API["cec_parse"](
                text=normalized_text,
                language=normalized_language,
                domain=normalized_domain,
            )
        )
    except Exception as exc:
        return _error_result(f"cec_parse failed: {exc}", {"formula": None})
    normalized = _normalize_result(payload)
    normalized.setdefault("language_used", normalized_language)
    normalized.setdefault("domain", normalized_domain)
    return normalized


async def cec_validate_formula(formula: str) -> Dict[str, Any]:
    """Validate a CEC/DCEC formula string."""
    normalized_formula = _normalize_non_empty_string(formula, "formula")
    if normalized_formula is None:
        return _error_result(
            "'formula' is required.",
            {"valid": False, "errors": ["'formula' is required."], "warnings": []},
        )
    try:
        payload = await _await_maybe(_API["cec_validate_formula"](formula=normalized_formula))
    except Exception as exc:
        return _error_result(f"cec_validate_formula failed: {exc}", {"formula": normalized_formula, "valid": False})
    normalized = _normalize_result(payload)
    normalized.setdefault("formula", normalized_formula)
    normalized.setdefault("valid", False)
    normalized.setdefault("errors", [])
    normalized.setdefault("warnings", [])
    return normalized


async def cec_analyze_formula(formula: str) -> Dict[str, Any]:
    """Analyze a CEC/DCEC formula structure."""
    normalized_formula = _normalize_non_empty_string(formula, "formula")
    if normalized_formula is None:
        return _error_result("'formula' is required.")
    try:
        payload = await _await_maybe(_API["cec_analyze_formula"](formula=normalized_formula))
    except Exception as exc:
        return _error_result(f"cec_analyze_formula failed: {exc}", {"formula": normalized_formula})
    normalized = _normalize_result(payload)
    normalized.setdefault("formula", normalized_formula)
    return normalized


async def cec_formula_complexity(formula: str) -> Dict[str, Any]:
    """Return a coarse complexity classification for a CEC formula."""
    normalized_formula = _normalize_non_empty_string(formula, "formula")
    if normalized_formula is None:
        return _error_result("'formula' is required.")
    try:
        payload = await _await_maybe(_API["cec_formula_complexity"](formula=normalized_formula))
    except Exception as exc:
        return _error_result(f"cec_formula_complexity failed: {exc}", {"formula": normalized_formula})
    normalized = _normalize_result(payload)
    normalized.setdefault("formula", normalized_formula)
    return normalized


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
                "format": {"type": "string", "minLength": 1, "default": "symbolic"},
                "language": {"type": "string", "minLength": 1, "default": "en"},
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
                "source_format": {"type": "string", "minLength": 1, "default": "tdfol"},
                "target_format": {"type": "string", "minLength": 1, "default": "fol"},
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
                "strategy": {"type": "string", "minLength": 1, "default": "auto"},
                "timeout_ms": {"type": "integer", "minimum": 1, "default": 5000},
                "max_depth": {"type": "integer", "minimum": 1, "default": 10},
                "include_proof_steps": {"type": "boolean", "default": True},
            },
            "required": ["formula"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "logic-tools"],
    )

    manager.register_tool(
        category="logic_tools",
        name="tdfol_kb_add_axiom",
        func=tdfol_kb_add_axiom,
        description="Add an axiom to the TDFOL knowledge base.",
        input_schema={
            "type": "object",
            "properties": {
                "formula": {"type": "string", "minLength": 1},
            },
            "required": ["formula"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "logic-tools"],
    )

    manager.register_tool(
        category="logic_tools",
        name="tdfol_kb_add_theorem",
        func=tdfol_kb_add_theorem,
        description="Add a theorem to the TDFOL knowledge base.",
        input_schema={
            "type": "object",
            "properties": {
                "formula": {"type": "string", "minLength": 1},
            },
            "required": ["formula"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "logic-tools"],
    )

    manager.register_tool(
        category="logic_tools",
        name="tdfol_kb_query",
        func=tdfol_kb_query,
        description="Query TDFOL knowledge-base contents and statistics.",
        input_schema={"type": "object", "properties": {}, "required": []},
        runtime="fastapi",
        tags=["native", "mcpp", "logic-tools"],
    )

    manager.register_tool(
        category="logic_tools",
        name="tdfol_kb_export",
        func=tdfol_kb_export,
        description="Export the TDFOL knowledge base.",
        input_schema={
            "type": "object",
            "properties": {
                "export_format": {
                    "type": "string",
                    "enum": ["json", "smt2", "tptp"],
                    "default": "json",
                },
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "logic-tools"],
    )

    manager.register_tool(
        category="logic_tools",
        name="cec_prove",
        func=cec_prove,
        description="Prove a CEC/DCEC theorem.",
        input_schema={
            "type": "object",
            "properties": {
                "goal": {"type": "string", "minLength": 1},
                "axioms": {"type": ["array", "null"], "items": {"type": "string", "minLength": 1}},
                "strategy": {"type": "string", "minLength": 1, "default": "auto"},
                "timeout": {"type": "integer", "minimum": 1, "default": 30},
            },
            "required": ["goal"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "logic-tools"],
    )

    manager.register_tool(
        category="logic_tools",
        name="cec_check_theorem",
        func=cec_check_theorem,
        description="Check whether a CEC formula is a theorem.",
        input_schema={
            "type": "object",
            "properties": {
                "formula": {"type": "string", "minLength": 1},
                "axioms": {"type": ["array", "null"], "items": {"type": "string", "minLength": 1}},
            },
            "required": ["formula"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "logic-tools"],
    )

    manager.register_tool(
        category="logic_tools",
        name="cec_parse",
        func=cec_parse,
        description="Parse natural language text into a CEC/DCEC formula.",
        input_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string", "minLength": 1},
                "language": {"type": "string", "minLength": 1, "default": "en"},
                "domain": {"type": "string", "minLength": 1, "default": "general"},
            },
            "required": ["text"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "logic-tools"],
    )

    manager.register_tool(
        category="logic_tools",
        name="cec_validate_formula",
        func=cec_validate_formula,
        description="Validate a CEC/DCEC formula string.",
        input_schema={
            "type": "object",
            "properties": {
                "formula": {"type": "string", "minLength": 1},
            },
            "required": ["formula"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "logic-tools"],
    )

    manager.register_tool(
        category="logic_tools",
        name="cec_analyze_formula",
        func=cec_analyze_formula,
        description="Analyze a CEC/DCEC formula structure.",
        input_schema={
            "type": "object",
            "properties": {
                "formula": {"type": "string", "minLength": 1},
            },
            "required": ["formula"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "logic-tools"],
    )

    manager.register_tool(
        category="logic_tools",
        name="cec_formula_complexity",
        func=cec_formula_complexity,
        description="Classify CEC/DCEC formula complexity.",
        input_schema={
            "type": "object",
            "properties": {
                "formula": {"type": "string", "minLength": 1},
            },
            "required": ["formula"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "logic-tools"],
    )
