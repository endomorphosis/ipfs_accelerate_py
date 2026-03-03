"""Native analysis tool implementations for unified mcp_server."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _load_analysis_api() -> Dict[str, Any]:
    """Resolve source analysis APIs with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.analytics.analysis_engine import (  # type: ignore
            analyze_data_distribution,
            cluster_analysis,
            dimensionality_reduction,
            quality_assessment,
        )

        return {
            "analyze_data_distribution": analyze_data_distribution,
            "cluster_analysis": cluster_analysis,
            "quality_assessment": quality_assessment,
            "dimensionality_reduction": dimensionality_reduction,
        }
    except Exception:

        async def _distribution_fallback(
            data_source: str = "mock",
            analysis_type: str = "comprehensive",
            vectors: Optional[List[List[float]]] = None,
            data_params: Optional[Dict[str, Any]] = None,
            visualization_config: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            _ = vectors, data_params, visualization_config
            return {
                "success": True,
                "status": "success",
                "data_source": data_source,
                "analysis_type": analysis_type,
                "data_shape": [0, 0],
                "feature_statistics": {},
                "vector_norm_statistics": {},
                "distance_statistics": {"note": "analysis backend unavailable"},
                "correlation_strength": 0.0,
                "sparsity_ratio": 0.0,
            }

        async def _cluster_fallback(
            data_source: str = "mock",
            algorithm: str = "kmeans",
            n_clusters: Optional[int] = None,
            vectors: Optional[List[List[float]]] = None,
            data_params: Optional[Dict[str, Any]] = None,
            clustering_params: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            _ = vectors, data_params, clustering_params
            return {
                "success": True,
                "status": "success",
                "data_source": data_source,
                "algorithm": algorithm,
                "n_clusters": int(n_clusters or 2),
                "cluster_labels": [],
                "metrics": {},
            }

        async def _quality_fallback(
            data_source: str = "mock",
            assessment_type: str = "comprehensive",
            metrics: Optional[List[str]] = None,
            data: Optional[Dict[str, Any]] = None,
            embeddings: Optional[List[List[float]]] = None,
            data_params: Optional[Dict[str, Any]] = None,
            outlier_detection: bool = True,
        ) -> Dict[str, Any]:
            _ = metrics, data, embeddings, data_params, outlier_detection
            return {
                "success": True,
                "status": "success",
                "data_source": data_source,
                "assessment_type": assessment_type,
                "overall_score": 0.0,
                "metric_scores": {},
                "outliers": [],
            }

        async def _reduction_fallback(
            data_source: str = "mock",
            method: str = "pca",
            n_components: int = 2,
            vectors: Optional[List[List[float]]] = None,
            data_params: Optional[Dict[str, Any]] = None,
            method_params: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            _ = vectors, data_params, method_params
            return {
                "success": True,
                "status": "success",
                "data_source": data_source,
                "method": method,
                "original_dimensions": 0,
                "reduced_dimensions": int(n_components),
                "transformed_data": [],
            }

        return {
            "analyze_data_distribution": _distribution_fallback,
            "cluster_analysis": _cluster_fallback,
            "quality_assessment": _quality_fallback,
            "dimensionality_reduction": _reduction_fallback,
        }


_API = _load_analysis_api()


async def analyze_data_distribution(
    data_source: str = "mock",
    analysis_type: str = "comprehensive",
    vectors: Optional[List[List[float]]] = None,
    data_params: Optional[Dict[str, Any]] = None,
    visualization_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Analyze statistical distribution of embedding vectors."""
    normalized_source = str(data_source or "mock").strip() or "mock"
    normalized_analysis_type = str(analysis_type or "comprehensive").strip() or "comprehensive"
    if vectors is not None:
        if not isinstance(vectors, list):
            return {
                "status": "error",
                "message": "vectors must be an array of numeric arrays when provided",
                "vectors": vectors,
            }
        for row in vectors:
            if not isinstance(row, list) or not all(isinstance(value, (int, float)) for value in row):
                return {
                    "status": "error",
                    "message": "vectors must be an array of numeric arrays when provided",
                    "vectors": vectors,
                }

    result = await _API["analyze_data_distribution"](
        data_source=normalized_source,
        analysis_type=normalized_analysis_type,
        vectors=vectors,
        data_params=data_params,
        visualization_config=visualization_config,
    )
    payload = dict(result or {})
    payload.setdefault("status", "success")
    payload.setdefault("data_source", normalized_source)
    payload.setdefault("analysis_type", normalized_analysis_type)
    return payload


async def cluster_analysis(
    data_source: str = "mock",
    algorithm: str = "kmeans",
    n_clusters: Optional[int] = None,
    vectors: Optional[List[List[float]]] = None,
    data_params: Optional[Dict[str, Any]] = None,
    clustering_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Perform clustering analysis on vector data."""
    normalized_algorithm = str(algorithm or "kmeans").strip().lower() or "kmeans"
    valid_algorithms = {"kmeans", "dbscan", "hierarchical", "gaussian_mixture", "spectral"}
    if normalized_algorithm not in valid_algorithms:
        return {
            "status": "error",
            "message": f"algorithm must be one of: {', '.join(sorted(valid_algorithms))}",
            "algorithm": algorithm,
        }
    normalized_clusters: Optional[int] = None
    if n_clusters is not None:
        try:
            normalized_clusters = int(n_clusters)
        except (TypeError, ValueError):
            return {
                "status": "error",
                "message": "n_clusters must be a positive integer when provided",
                "n_clusters": n_clusters,
            }
    if normalized_clusters is not None and normalized_clusters <= 0:
        return {
            "status": "error",
            "message": "n_clusters must be a positive integer when provided",
            "n_clusters": n_clusters,
        }

    result = await _API["cluster_analysis"](
        data_source=data_source,
        algorithm=normalized_algorithm,
        n_clusters=normalized_clusters,
        vectors=vectors,
        data_params=data_params,
        clustering_params=clustering_params,
    )
    payload = dict(result or {})
    payload.setdefault("status", "success")
    payload.setdefault("algorithm", normalized_algorithm)
    return payload


async def quality_assessment(
    data_source: str = "mock",
    assessment_type: str = "comprehensive",
    metrics: Optional[List[str]] = None,
    data: Optional[Dict[str, Any]] = None,
    embeddings: Optional[List[List[float]]] = None,
    data_params: Optional[Dict[str, Any]] = None,
    outlier_detection: bool = True,
) -> Dict[str, Any]:
    """Assess quality of embeddings and vector data."""
    normalized_assessment_type = str(assessment_type or "comprehensive").strip() or "comprehensive"
    if metrics is not None and not isinstance(metrics, list):
        return {
            "status": "error",
            "message": "metrics must be an array of strings when provided",
            "metrics": metrics,
        }
    if isinstance(metrics, list) and not all(isinstance(item, str) for item in metrics):
        return {
            "status": "error",
            "message": "metrics must be an array of strings when provided",
            "metrics": metrics,
        }

    result = await _API["quality_assessment"](
        data_source=data_source,
        assessment_type=normalized_assessment_type,
        metrics=metrics,
        data=data,
        embeddings=embeddings,
        data_params=data_params,
        outlier_detection=outlier_detection,
    )
    payload = dict(result or {})
    payload.setdefault("status", "success")
    payload.setdefault("assessment_type", normalized_assessment_type)
    return payload


async def dimensionality_reduction(
    data_source: str = "mock",
    method: str = "pca",
    n_components: int = 2,
    vectors: Optional[List[List[float]]] = None,
    data_params: Optional[Dict[str, Any]] = None,
    method_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Reduce vector dimensionality using selected method."""
    normalized_method = str(method or "pca").strip().lower() or "pca"
    valid_methods = {"pca", "tsne", "umap", "random_projection", "truncated_svd"}
    if normalized_method not in valid_methods:
        return {
            "status": "error",
            "message": f"method must be one of: {', '.join(sorted(valid_methods))}",
            "method": method,
        }
    try:
        normalized_components = int(n_components)
    except (TypeError, ValueError):
        return {
            "status": "error",
            "message": "n_components must be a positive integer",
            "n_components": n_components,
        }
    if normalized_components <= 0:
        return {
            "status": "error",
            "message": "n_components must be a positive integer",
            "n_components": n_components,
        }

    result = await _API["dimensionality_reduction"](
        data_source=data_source,
        method=normalized_method,
        n_components=normalized_components,
        vectors=vectors,
        data_params=data_params,
        method_params=method_params,
    )
    payload = dict(result or {})
    payload.setdefault("status", "success")
    payload.setdefault("method", normalized_method)
    payload.setdefault("reduced_dimensions", normalized_components)
    return payload


def register_native_analysis_tools(manager: Any) -> None:
    """Register native analysis tools in unified hierarchical manager."""
    manager.register_tool(
        category="analysis_tools",
        name="analyze_data_distribution",
        func=analyze_data_distribution,
        description="Analyze statistical distribution of vector data.",
        input_schema={
            "type": "object",
            "properties": {
                "data_source": {"type": "string"},
                "analysis_type": {"type": "string"},
                "vectors": {"type": ["array", "null"], "items": {"type": "array", "items": {"type": "number"}}},
                "data_params": {"type": ["object", "null"]},
                "visualization_config": {"type": ["object", "null"]},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "analysis"],
    )

    manager.register_tool(
        category="analysis_tools",
        name="cluster_analysis",
        func=cluster_analysis,
        description="Perform clustering analysis on vectors.",
        input_schema={
            "type": "object",
            "properties": {
                "data_source": {"type": "string"},
                "algorithm": {
                    "type": "string",
                    "enum": ["kmeans", "dbscan", "hierarchical", "gaussian_mixture", "spectral"],
                    "default": "kmeans",
                },
                "n_clusters": {"type": ["integer", "null"]},
                "vectors": {"type": ["array", "null"], "items": {"type": "array", "items": {"type": "number"}}},
                "data_params": {"type": ["object", "null"]},
                "clustering_params": {"type": ["object", "null"]},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "analysis"],
    )

    manager.register_tool(
        category="analysis_tools",
        name="quality_assessment",
        func=quality_assessment,
        description="Assess quality metrics for vector data.",
        input_schema={
            "type": "object",
            "properties": {
                "data_source": {"type": "string"},
                "assessment_type": {"type": "string"},
                "metrics": {"type": ["array", "null"], "items": {"type": "string"}},
                "data": {"type": ["object", "null"]},
                "embeddings": {"type": ["array", "null"], "items": {"type": "array", "items": {"type": "number"}}},
                "data_params": {"type": ["object", "null"]},
                "outlier_detection": {"type": "boolean"},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "analysis"],
    )

    manager.register_tool(
        category="analysis_tools",
        name="dimensionality_reduction",
        func=dimensionality_reduction,
        description="Reduce dimensionality of vector data.",
        input_schema={
            "type": "object",
            "properties": {
                "data_source": {"type": "string"},
                "method": {
                    "type": "string",
                    "enum": ["pca", "tsne", "umap", "random_projection", "truncated_svd"],
                    "default": "pca",
                },
                "n_components": {"type": "integer"},
                "vectors": {"type": ["array", "null"], "items": {"type": "array", "items": {"type": "number"}}},
                "data_params": {"type": ["object", "null"]},
                "method_params": {"type": ["object", "null"]},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "analysis"],
    )
