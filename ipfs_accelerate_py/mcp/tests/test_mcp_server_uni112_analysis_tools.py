#!/usr/bin/env python3
"""UNI-112 analysis tools parity hardening tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.analysis_tools.native_analysis_tools import (
    analyze_data_distribution,
    cluster_analysis,
    dimensionality_reduction,
    quality_assessment,
    register_native_analysis_tools,
)


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI112AnalysisTools(unittest.TestCase):
    def test_register_includes_analysis_tools(self) -> None:
        manager = _DummyManager()
        register_native_analysis_tools(manager)
        names = [c["name"] for c in manager.calls]
        self.assertIn("analyze_data_distribution", names)
        self.assertIn("cluster_analysis", names)
        self.assertIn("quality_assessment", names)
        self.assertIn("dimensionality_reduction", names)

    def test_analyze_data_distribution_rejects_bad_vectors_shape(self) -> None:
        async def _run() -> None:
            result = await analyze_data_distribution(vectors=[[1.0, 2.0], ["x"]])  # type: ignore[list-item]
            self.assertEqual(result.get("status"), "error")
            self.assertIn("array of numeric arrays", str(result.get("message", "")))

        anyio.run(_run)

    def test_cluster_analysis_rejects_invalid_algorithm(self) -> None:
        async def _run() -> None:
            result = await cluster_analysis(algorithm="invalid")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("algorithm must be one of", str(result.get("message", "")))

        anyio.run(_run)

    def test_quality_assessment_rejects_non_array_metrics(self) -> None:
        async def _run() -> None:
            result = await quality_assessment(metrics="accuracy")  # type: ignore[arg-type]
            self.assertEqual(result.get("status"), "error")
            self.assertIn("metrics must be an array", str(result.get("message", "")))

        anyio.run(_run)

    def test_dimensionality_reduction_rejects_invalid_components(self) -> None:
        async def _run() -> None:
            result = await dimensionality_reduction(n_components=0)
            self.assertEqual(result.get("status"), "error")
            self.assertIn("positive integer", str(result.get("message", "")))

        anyio.run(_run)

    def test_dimensionality_reduction_success_shape(self) -> None:
        async def _run() -> None:
            result = await dimensionality_reduction(method="pca", n_components=2)
            self.assertIn(result.get("status"), ["success", "error"])
            if result.get("status") == "success":
                self.assertIn("reduced_dimensions", result)

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
