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


async def get_tdfol_metrics() -> Dict[str, Any]:
    """Get current TDFOL performance metrics."""
    result = _API["get_tdfol_metrics"]()
    if hasattr(result, "__await__"):
        return await result
    return result


async def profile_tdfol_operation(
    formula_str: str,
    kb_formulas: Optional[List[str]] = None,
    runs: int = 10,
    strategy: Optional[str] = None,
) -> Dict[str, Any]:
    """Profile a TDFOL proving operation."""
    result = _API["profile_tdfol_operation"](formula_str, kb_formulas, runs, strategy)
    if hasattr(result, "__await__"):
        return await result
    return result


async def generate_tdfol_dashboard(
    output_path: Optional[str] = None,
    include_profiling: bool = False,
) -> Dict[str, Any]:
    """Generate TDFOL performance dashboard HTML."""
    result = _API["generate_tdfol_dashboard"](output_path, include_profiling)
    if hasattr(result, "__await__"):
        return await result
    return result


async def export_tdfol_statistics(
    format: str = "json",
    include_raw_data: bool = False,
) -> Dict[str, Any]:
    """Export TDFOL performance statistics."""
    result = _API["export_tdfol_statistics"](format, include_raw_data)
    if hasattr(result, "__await__"):
        return await result
    return result


async def get_tdfol_profiler_report(
    report_format: str = "text",
    top_n: int = 20,
) -> Dict[str, Any]:
    """Get detailed TDFOL profiler report."""
    result = _API["get_tdfol_profiler_report"](report_format, top_n)
    if hasattr(result, "__await__"):
        return await result
    return result


async def compare_tdfol_strategies(
    formula_str: str,
    strategies: Optional[List[str]] = None,
    kb_formulas: Optional[List[str]] = None,
    runs_per_strategy: int = 10,
) -> Dict[str, Any]:
    """Compare TDFOL proving strategies performance."""
    result = _API["compare_tdfol_strategies"](formula_str, strategies, kb_formulas, runs_per_strategy)
    if hasattr(result, "__await__"):
        return await result
    return result


async def check_tdfol_performance_regression(
    baseline_path: Optional[str] = None,
    threshold_percent: float = 10.0,
) -> Dict[str, Any]:
    """Check for TDFOL performance regressions."""
    result = _API["check_tdfol_performance_regression"](baseline_path, threshold_percent)
    if hasattr(result, "__await__"):
        return await result
    return result


async def reset_tdfol_metrics() -> Dict[str, Any]:
    """Reset TDFOL performance metrics and collectors."""
    result = _API["reset_tdfol_metrics"]()
    if hasattr(result, "__await__"):
        return await result
    return result


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
                "runs": {"type": "integer"},
                "strategy": {"type": ["string", "null"]},
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
                "include_profiling": {"type": "boolean"},
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
                "format": {"type": "string"},
                "include_raw_data": {"type": "boolean"},
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
                "report_format": {"type": "string"},
                "top_n": {"type": "integer"},
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
                "strategies": {"type": ["array", "null"], "items": {"type": "string"}},
                "kb_formulas": {"type": ["array", "null"], "items": {"type": "string"}},
                "runs_per_strategy": {"type": "integer"},
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
                "threshold_percent": {"type": "number"},
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
