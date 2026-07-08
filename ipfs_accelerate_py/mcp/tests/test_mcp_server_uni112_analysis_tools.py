#!/usr/bin/env python3
"""UNI-112 analysis tools parity hardening tests."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.mcp_server.tools.analysis_tools import native_analysis_tools
from ipfs_accelerate_py.mcp_server.tools.analysis_tools.native_analysis_tools import (
    analyze_data_distribution,
    cluster_analysis,
    detect_outliers,
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
        self.assertIn("detect_outliers", names)
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

    def test_cluster_analysis_accepts_source_algorithm_and_rejects_bad_cluster_type(self) -> None:
        async def _run() -> None:
            spectral_result = await cluster_analysis(algorithm="spectral")
            self.assertIn(spectral_result.get("status"), ["success", "error"])

            bad_clusters = await cluster_analysis(algorithm="kmeans", n_clusters="bad")  # type: ignore[arg-type]
            self.assertEqual(bad_clusters.get("status"), "error")
            self.assertIn("n_clusters must be a positive integer", str(bad_clusters.get("message", "")))

        anyio.run(_run)

    def test_quality_assessment_rejects_non_array_metrics(self) -> None:
        async def _run() -> None:
            result = await quality_assessment(metrics="accuracy")  # type: ignore[arg-type]
            self.assertEqual(result.get("status"), "error")
            self.assertIn("metrics must be an array", str(result.get("message", "")))

        anyio.run(_run)

    def test_detect_outliers_rejects_invalid_data_shape(self) -> None:
        async def _run() -> None:
            result = await detect_outliers(data=[[0.1, 0.2], ["bad"]])  # type: ignore[list-item]
            self.assertEqual(result.get("status"), "error")
            self.assertIn("array of numeric arrays", str(result.get("message", "")))

        anyio.run(_run)

    def test_detect_outliers_rejects_invalid_threshold(self) -> None:
        async def _run() -> None:
            result = await detect_outliers(data=[[0.1, 0.2], [0.2, 0.3]], threshold=0)
            self.assertEqual(result.get("status"), "error")
            self.assertIn("threshold must be a positive number", str(result.get("message", "")))

        anyio.run(_run)

    def test_detect_outliers_success_shape(self) -> None:
        async def _run() -> None:
            result = await detect_outliers(data=[[0.1, 0.2], [0.2, 0.3], [10.0, 12.0]], threshold=1.5)
            self.assertEqual(result.get("status"), "success")
            self.assertIn("outlier_indices", result)
            self.assertIn("outlier_count", result)

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

    def test_dimensionality_reduction_accepts_source_method_and_rejects_bad_type(self) -> None:
        async def _run() -> None:
            source_method = await dimensionality_reduction(method="truncated_svd", n_components=2)
            self.assertIn(source_method.get("status"), ["success", "error"])

            bad_components = await dimensionality_reduction(method="pca", n_components="bad")  # type: ignore[arg-type]
            self.assertEqual(bad_components.get("status"), "error")
            self.assertIn("n_components must be a positive integer", str(bad_components.get("message", "")))

        anyio.run(_run)

    def test_analysis_wrappers_infer_error_status_from_contradictory_delegate_payloads(self) -> None:
        async def _contradictory_failure(**_: object) -> dict:
            return {"status": "success", "success": False, "error": "delegate failed"}

        async def _run() -> None:
            with patch.dict(
                native_analysis_tools._API,
                {
                    "analyze_data_distribution": _contradictory_failure,
                    "cluster_analysis": _contradictory_failure,
                    "quality_assessment": _contradictory_failure,
                    "detect_outliers": _contradictory_failure,
                    "dimensionality_reduction": _contradictory_failure,
                },
                clear=False,
            ):
                distribution = await analyze_data_distribution()
                clusters = await cluster_analysis()
                quality = await quality_assessment()
                outliers = await detect_outliers(data=[[0.1, 0.2], [0.2, 0.3]])
                reduced = await dimensionality_reduction()

            self.assertEqual(distribution.get("status"), "error")
            self.assertEqual(distribution.get("error"), "delegate failed")

            self.assertEqual(clusters.get("status"), "error")
            self.assertEqual(clusters.get("error"), "delegate failed")

            self.assertEqual(quality.get("status"), "error")
            self.assertEqual(quality.get("error"), "delegate failed")

            self.assertEqual(outliers.get("status"), "error")
            self.assertEqual(outliers.get("error"), "delegate failed")

            self.assertEqual(reduced.get("status"), "error")
            self.assertEqual(reduced.get("error"), "delegate failed")

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
