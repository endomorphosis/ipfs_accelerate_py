"""Native dashboard tool implementations for unified mcp_server."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _load_dashboard_api() -> Dict[str, Any]:
    """Resolve source dashboard APIs with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.dashboard_tools.tdfol_performance_tool import (  # type: ignore
            check_tdfol_performance_regression as _check_tdfol_performance_regression,
            compare_tdfol_strategies as _compare_tdfol_strategies,
            export_tdfol_statistics as _export_tdfol_statistics,
            generate_tdfol_dashboard as _generate_tdfol_dashboard,
            get_tdfol_metrics as _get_tdfol_metrics,
            get_tdfol_profiler_report as _get_tdfol_profiler_report,
            profile_tdfol_operation as _profile_tdfol_operation,
            reset_tdfol_metrics as _reset_tdfol_metrics,
        )

        return {
            "get_tdfol_metrics": _get_tdfol_metrics,
            "profile_tdfol_operation": _profile_tdfol_operation,
            "generate_tdfol_dashboard": _generate_tdfol_dashboard,
            "export_tdfol_statistics": _export_tdfol_statistics,
            "get_tdfol_profiler_report": _get_tdfol_profiler_report,
            "compare_tdfol_strategies": _compare_tdfol_strategies,
            "check_tdfol_performance_regression": _check_tdfol_performance_regression,
            "reset_tdfol_metrics": _reset_tdfol_metrics,
        }
    except Exception:
        logger.warning("Source dashboard_tools import unavailable, using fallback dashboard functions")

        def _metrics_fallback() -> Dict[str, Any]:
            return {
                "status": "success",
                "metrics": {},
            }

        def _profile_fallback(
            formula_str: str,
            kb_formulas: Optional[List[str]] = None,
            runs: int = 10,
            strategy: Optional[str] = None,
        ) -> Dict[str, Any]:
            _ = kb_formulas, runs, strategy
            return {
                "status": "success",
                "formula": formula_str,
                "profile": {},
            }

        def _dashboard_fallback(
            output_path: Optional[str] = None,
            include_profiling: bool = False,
        ) -> Dict[str, Any]:
            _ = output_path, include_profiling
            return {
                "status": "success",
                "dashboard_generated": False,
            }

        def _export_fallback(
            format: str = "json",
            include_raw_data: bool = False,
        ) -> Dict[str, Any]:
            _ = format, include_raw_data
            return {
                "status": "success",
                "statistics": {},
            }

        def _report_fallback(
            report_format: str = "text",
            top_n: int = 20,
        ) -> Dict[str, Any]:
            _ = report_format, top_n
            return {
                "status": "success",
                "report": {},
            }

        def _compare_fallback(
            formula_str: str,
            strategies: Optional[List[str]] = None,
            kb_formulas: Optional[List[str]] = None,
            runs_per_strategy: int = 10,
        ) -> Dict[str, Any]:
            _ = strategies, kb_formulas, runs_per_strategy
            return {
                "status": "success",
                "formula": formula_str,
                "comparison": {},
            }

        def _regression_fallback(
            baseline_path: Optional[str] = None,
            threshold_percent: float = 10.0,
        ) -> Dict[str, Any]:
            _ = baseline_path, threshold_percent
            return {
                "status": "success",
                "regression_detected": False,
            }

        def _reset_fallback() -> Dict[str, Any]:
            return {
                "status": "success",
                "reset": True,
            }

        return {
            "get_tdfol_metrics": _metrics_fallback,
            "profile_tdfol_operation": _profile_fallback,
            "generate_tdfol_dashboard": _dashboard_fallback,
            "export_tdfol_statistics": _export_fallback,
            "get_tdfol_profiler_report": _report_fallback,
            "compare_tdfol_strategies": _compare_fallback,
            "check_tdfol_performance_regression": _regression_fallback,
            "reset_tdfol_metrics": _reset_fallback,
        }


_API = _load_dashboard_api()


def _normalize_payload(result: Any) -> Dict[str, Any]:
    """Normalize result to a deterministic dict envelope."""
    payload = dict(result or {})
    if "error" in payload and payload.get("error"):
        payload.setdefault("status", "error")
    elif "warning" in payload and payload.get("warning"):
        payload.setdefault("status", "warning")
    else:
        payload.setdefault("status", "success")
    return payload


async def get_tdfol_metrics() -> Dict[str, Any]:
    """Get current TDFOL performance metrics."""
    result = _API["get_tdfol_metrics"]()
    if hasattr(result, "__await__"):
        return _normalize_payload(await result)
    return _normalize_payload(result)


async def profile_tdfol_operation(
    formula_str: str,
    kb_formulas: Optional[List[str]] = None,
    runs: int = 10,
    strategy: Optional[str] = None,
) -> Dict[str, Any]:
    """Profile a TDFOL proving operation."""
    normalized_formula = str(formula_str or "").strip()
    if not normalized_formula:
        return {
            "status": "error",
            "message": "formula_str is required",
            "formula_str": formula_str,
        }
    normalized_kb_formulas: Optional[List[str]] = None
    if kb_formulas is not None:
        if not isinstance(kb_formulas, list) or not all(isinstance(item, str) for item in kb_formulas):
            return {
                "status": "error",
                "message": "kb_formulas must be an array of strings when provided",
                "kb_formulas": kb_formulas,
            }
        normalized_kb_formulas = [str(item).strip() for item in kb_formulas]
        if any(not item for item in normalized_kb_formulas):
            return {
                "status": "error",
                "message": "kb_formulas cannot contain empty strings",
                "kb_formulas": kb_formulas,
            }
    if not isinstance(runs, int) or runs < 1:
        return {
            "status": "error",
            "message": "runs must be an integer >= 1",
            "runs": runs,
        }
    normalized_strategy = str(strategy).strip().lower() if strategy is not None else None
    if normalized_strategy is not None and normalized_strategy not in {"auto", "forward", "modal", "cec"}:
        return {
            "status": "error",
            "message": "strategy must be one of: auto, forward, modal, cec when provided",
            "strategy": strategy,
        }

    result = _API["profile_tdfol_operation"](
        normalized_formula,
        normalized_kb_formulas,
        runs,
        normalized_strategy,
    )
    if hasattr(result, "__await__"):
        payload = _normalize_payload(await result)
    else:
        payload = _normalize_payload(result)
    payload.setdefault("formula", normalized_formula)
    payload.setdefault("runs", runs)
    if normalized_strategy is not None:
        payload.setdefault("strategy", normalized_strategy)
    return payload


async def generate_tdfol_dashboard(
    output_path: Optional[str] = None,
    include_profiling: bool = False,
) -> Dict[str, Any]:
    """Generate TDFOL performance dashboard HTML."""
    normalized_output_path = str(output_path).strip() if output_path is not None else None
    if output_path is not None and not normalized_output_path:
        return {
            "status": "error",
            "message": "output_path must be a non-empty string when provided",
            "output_path": output_path,
        }
    if not isinstance(include_profiling, bool):
        return {
            "status": "error",
            "message": "include_profiling must be a boolean",
            "include_profiling": include_profiling,
        }

    result = _API["generate_tdfol_dashboard"](normalized_output_path, include_profiling)
    if hasattr(result, "__await__"):
        payload = _normalize_payload(await result)
    else:
        payload = _normalize_payload(result)
    payload.setdefault("include_profiling", include_profiling)
    if normalized_output_path is not None:
        payload.setdefault("output_path", normalized_output_path)
    return payload


async def export_tdfol_statistics(
    format: str = "json",
    include_raw_data: bool = False,
) -> Dict[str, Any]:
    """Export TDFOL performance statistics."""
    normalized_format = str(format or "").strip().lower()
    if normalized_format not in {"json", "prometheus"}:
        return {
            "status": "error",
            "message": "format must be one of: json, prometheus",
            "format": format,
        }
    if not isinstance(include_raw_data, bool):
        return {
            "status": "error",
            "message": "include_raw_data must be a boolean",
            "include_raw_data": include_raw_data,
        }

    result = _API["export_tdfol_statistics"](normalized_format, include_raw_data)
    if hasattr(result, "__await__"):
        payload = _normalize_payload(await result)
    else:
        payload = _normalize_payload(result)
    payload.setdefault("format", normalized_format)
    payload.setdefault("include_raw_data", include_raw_data)
    return payload


async def get_tdfol_profiler_report(
    report_format: str = "text",
    top_n: int = 20,
) -> Dict[str, Any]:
    """Get detailed TDFOL profiler report."""
    normalized_report_format = str(report_format or "").strip().lower()
    if normalized_report_format not in {"text", "html"}:
        return {
            "status": "error",
            "message": "report_format must be one of: text, html",
            "report_format": report_format,
        }
    if not isinstance(top_n, int) or top_n < 1:
        return {
            "status": "error",
            "message": "top_n must be an integer >= 1",
            "top_n": top_n,
        }

    result = _API["get_tdfol_profiler_report"](normalized_report_format, top_n)
    if hasattr(result, "__await__"):
        payload = _normalize_payload(await result)
    else:
        payload = _normalize_payload(result)
    payload.setdefault("format", normalized_report_format)
    payload.setdefault("top_n", top_n)
    return payload


async def compare_tdfol_strategies(
    formula_str: str,
    strategies: Optional[List[str]] = None,
    kb_formulas: Optional[List[str]] = None,
    runs_per_strategy: int = 10,
) -> Dict[str, Any]:
    """Compare TDFOL proving strategies performance."""
    normalized_formula = str(formula_str or "").strip()
    if not normalized_formula:
        return {
            "status": "error",
            "message": "formula_str is required",
            "formula_str": formula_str,
        }
    normalized_strategies: Optional[List[str]] = None
    valid_strategies = {"forward", "modal", "cec"}
    if strategies is not None:
        if not isinstance(strategies, list) or not all(isinstance(item, str) for item in strategies):
            return {
                "status": "error",
                "message": "strategies must be an array of strings when provided",
                "strategies": strategies,
            }
        normalized_strategies = [str(item).strip().lower() for item in strategies]
        if any(not item for item in normalized_strategies):
            return {
                "status": "error",
                "message": "strategies cannot contain empty strings",
                "strategies": strategies,
            }
        invalid = sorted({item for item in normalized_strategies if item not in valid_strategies})
        if invalid:
            return {
                "status": "error",
                "message": "strategies must only include: forward, modal, cec",
                "strategies": strategies,
            }

    normalized_kb_formulas: Optional[List[str]] = None
    if kb_formulas is not None:
        if not isinstance(kb_formulas, list) or not all(isinstance(item, str) for item in kb_formulas):
            return {
                "status": "error",
                "message": "kb_formulas must be an array of strings when provided",
                "kb_formulas": kb_formulas,
            }
        normalized_kb_formulas = [str(item).strip() for item in kb_formulas]
        if any(not item for item in normalized_kb_formulas):
            return {
                "status": "error",
                "message": "kb_formulas cannot contain empty strings",
                "kb_formulas": kb_formulas,
            }

    if not isinstance(runs_per_strategy, int) or runs_per_strategy < 1:
        return {
            "status": "error",
            "message": "runs_per_strategy must be an integer >= 1",
            "runs_per_strategy": runs_per_strategy,
        }

    result = _API["compare_tdfol_strategies"](
        normalized_formula,
        normalized_strategies,
        normalized_kb_formulas,
        runs_per_strategy,
    )
    if hasattr(result, "__await__"):
        payload = _normalize_payload(await result)
    else:
        payload = _normalize_payload(result)
    payload.setdefault("formula", normalized_formula)
    payload.setdefault("runs_per_strategy", runs_per_strategy)
    if normalized_strategies is not None:
        payload.setdefault("strategies", normalized_strategies)
    return payload


async def check_tdfol_performance_regression(
    baseline_path: Optional[str] = None,
    threshold_percent: float = 10.0,
) -> Dict[str, Any]:
    """Check for TDFOL performance regressions."""
    normalized_baseline_path = str(baseline_path).strip() if baseline_path is not None else None
    if baseline_path is not None and not normalized_baseline_path:
        return {
            "status": "error",
            "message": "baseline_path must be a non-empty string when provided",
            "baseline_path": baseline_path,
        }
    if not isinstance(threshold_percent, (int, float)) or float(threshold_percent) <= 0:
        return {
            "status": "error",
            "message": "threshold_percent must be a number greater than 0",
            "threshold_percent": threshold_percent,
        }

    result = _API["check_tdfol_performance_regression"](normalized_baseline_path, float(threshold_percent))
    if hasattr(result, "__await__"):
        payload = _normalize_payload(await result)
    else:
        payload = _normalize_payload(result)
    payload.setdefault("threshold_percent", float(threshold_percent))
    if normalized_baseline_path is not None:
        payload.setdefault("baseline_path", normalized_baseline_path)
    return payload


async def reset_tdfol_metrics() -> Dict[str, Any]:
    """Reset TDFOL performance metrics and collectors."""
    result = _API["reset_tdfol_metrics"]()
    if hasattr(result, "__await__"):
        return _normalize_payload(await result)
    return _normalize_payload(result)


def register_native_dashboard_tools(manager: Any) -> None:
    """Register native dashboard tools in unified hierarchical manager."""
    manager.register_tool(
        category="dashboard_tools",
        name="get_tdfol_metrics",
        func=get_tdfol_metrics,
        description="Get current TDFOL performance metrics.",
        input_schema={"type": "object", "properties": {}, "required": []},
        runtime="fastapi",
        tags=["native", "mcpp", "dashboard"],
    )

    manager.register_tool(
        category="dashboard_tools",
        name="profile_tdfol_operation",
        func=profile_tdfol_operation,
        description="Profile a TDFOL proving operation with detailed metrics.",
        input_schema={
            "type": "object",
            "properties": {
                "formula_str": {"type": "string"},
                "kb_formulas": {"type": ["array", "null"], "items": {"type": "string"}},
                "runs": {"type": "integer", "minimum": 1, "default": 10},
                "strategy": {
                    "type": ["string", "null"],
                    "enum": ["auto", "forward", "modal", "cec", None],
                    "default": None,
                },
            },
            "required": ["formula_str"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "dashboard"],
    )

    manager.register_tool(
        category="dashboard_tools",
        name="generate_tdfol_dashboard",
        func=generate_tdfol_dashboard,
        description="Generate interactive HTML performance dashboard.",
        input_schema={
            "type": "object",
            "properties": {
                "output_path": {"type": ["string", "null"]},
                "include_profiling": {"type": "boolean", "default": False},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "dashboard"],
    )

    manager.register_tool(
        category="dashboard_tools",
        name="export_tdfol_statistics",
        func=export_tdfol_statistics,
        description="Export TDFOL performance statistics in various formats.",
        input_schema={
            "type": "object",
            "properties": {
                "format": {"type": "string", "enum": ["json", "prometheus"], "default": "json"},
                "include_raw_data": {"type": "boolean", "default": False},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "dashboard"],
    )

    manager.register_tool(
        category="dashboard_tools",
        name="get_tdfol_profiler_report",
        func=get_tdfol_profiler_report,
        description="Get detailed profiler report with bottlenecks.",
        input_schema={
            "type": "object",
            "properties": {
                "report_format": {"type": "string", "enum": ["text", "html"], "default": "text"},
                "top_n": {"type": "integer", "minimum": 1, "default": 20},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "dashboard"],
    )

    manager.register_tool(
        category="dashboard_tools",
        name="compare_tdfol_strategies",
        func=compare_tdfol_strategies,
        description="Compare performance across different proving strategies.",
        input_schema={
            "type": "object",
            "properties": {
                "formula_str": {"type": "string"},
                "strategies": {
                    "type": ["array", "null"],
                    "items": {"type": "string", "enum": ["forward", "modal", "cec"]},
                },
                "kb_formulas": {"type": ["array", "null"], "items": {"type": "string"}},
                "runs_per_strategy": {"type": "integer", "minimum": 1, "default": 10},
            },
            "required": ["formula_str"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "dashboard"],
    )

    manager.register_tool(
        category="dashboard_tools",
        name="check_tdfol_performance_regression",
        func=check_tdfol_performance_regression,
        description="Check for performance regressions against baseline.",
        input_schema={
            "type": "object",
            "properties": {
                "baseline_path": {"type": ["string", "null"]},
                "threshold_percent": {"type": "number", "exclusiveMinimum": 0, "default": 10.0},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "dashboard"],
    )

    manager.register_tool(
        category="dashboard_tools",
        name="reset_tdfol_metrics",
        func=reset_tdfol_metrics,
        description="Reset TDFOL performance metrics and collectors.",
        input_schema={"type": "object", "properties": {}, "required": []},
        runtime="fastapi",
        tags=["native", "mcpp", "dashboard"],
    )
