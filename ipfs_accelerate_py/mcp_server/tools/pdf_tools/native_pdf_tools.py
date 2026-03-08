"""Native pdf-tools category implementations for unified mcp_server."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def _load_pdf_tools_api() -> Dict[str, Any]:
    """Resolve source pdf-tools APIs with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.pdf_tools import (  # type: ignore
            pdf_analyze_relationships as _pdf_analyze_relationships,
            pdf_batch_process as _pdf_batch_process,
            pdf_cross_document_analysis as _pdf_cross_document_analysis,
            pdf_extract_entities as _pdf_extract_entities,
            pdf_ingest_to_graphrag as _pdf_ingest_to_graphrag,
            pdf_optimize_for_llm as _pdf_optimize_for_llm,
            pdf_query_corpus as _pdf_query_corpus,
            pdf_query_knowledge_graph as _pdf_query_knowledge_graph,
        )

        return {
            "pdf_analyze_relationships": _pdf_analyze_relationships,
            "pdf_query_corpus": _pdf_query_corpus,
            "pdf_extract_entities": _pdf_extract_entities,
            "pdf_ingest_to_graphrag": _pdf_ingest_to_graphrag,
            "pdf_batch_process": _pdf_batch_process,
            "pdf_cross_document_analysis": _pdf_cross_document_analysis,
            "pdf_optimize_for_llm": _pdf_optimize_for_llm,
            "pdf_query_knowledge_graph": _pdf_query_knowledge_graph,
        }
    except Exception:
        logger.warning("Source pdf_tools import unavailable, using fallback pdf-tools functions")

        async def _relationships_fallback(
            document_id: str,
            analysis_type: str = "comprehensive",
            include_cross_document: bool = True,
            relationship_types: Optional[List[str]] = None,
            min_confidence: float = 0.6,
            max_relationships: int = 100,
        ) -> Dict[str, Any]:
            _ = analysis_type, include_cross_document, relationship_types, min_confidence, max_relationships
            return {
                "status": "error",
                "document_id": document_id,
                "message": "PDF relationship analysis backend unavailable",
            }

        async def _query_fallback(
            query: str,
            query_type: str = "hybrid",
            max_documents: int = 10,
            document_filters: Optional[Dict[str, Any]] = None,
            enable_reasoning: bool = True,
            include_sources: bool = True,
            confidence_threshold: float = 0.7,
        ) -> Dict[str, Any]:
            _ = query_type, max_documents, document_filters, enable_reasoning, include_sources, confidence_threshold
            return {
                "status": "error",
                "query": query,
                "message": "PDF query backend unavailable",
            }

        async def _knowledge_graph_fallback(
            graph_id: str,
            query: str,
            query_type: str = "sparql",
            max_results: int = 100,
            include_metadata: bool = True,
            return_subgraph: bool = False,
        ) -> Dict[str, Any]:
            _ = query_type, max_results, include_metadata, return_subgraph
            return {
                "status": "error",
                "graph_id": graph_id,
                "query": query,
                "message": "PDF knowledge graph backend unavailable",
            }

        async def _extract_fallback(
            pdf_source: Union[str, dict, None] = None,
            entity_types: Optional[List[str]] = None,
            extraction_method: str = "hybrid",
            confidence_threshold: float = 0.7,
            include_relationships: bool = True,
            context_window: int = 3,
            custom_patterns: Optional[Dict[str, str]] = None,
        ) -> Dict[str, Any]:
            _ = entity_types, extraction_method, confidence_threshold, include_relationships, context_window, custom_patterns
            return {
                "status": "error",
                "pdf_source": pdf_source,
                "message": "PDF entity extraction backend unavailable",
            }

        async def _ingest_fallback(
            pdf_source: Union[str, Dict[str, Any], None] = None,
            collection_name: str = "default",
            metadata: Optional[Dict[str, Any]] = None,
            chunk_strategy: str = "semantic",
            max_chunk_size: int = 4000,
            overlap_size: int = 200,
            extract_entities: bool = True,
            build_knowledge_graph: bool = True,
            store_embeddings: bool = True,
        ) -> Dict[str, Any]:
            _ = (
                collection_name,
                metadata,
                chunk_strategy,
                max_chunk_size,
                overlap_size,
                extract_entities,
                build_knowledge_graph,
                store_embeddings,
            )
            return {
                "status": "error",
                "pdf_source": pdf_source,
                "message": "PDF GraphRAG ingestion backend unavailable",
            }

        async def _batch_fallback(
            pdf_sources: Optional[List[Union[str, Dict[str, Any]]]] = None,
            batch_size: int = 5,
            parallel_workers: int = 3,
            enable_ocr: bool = True,
            target_llm: str = "gpt-4",
            chunk_strategy: str = "semantic",
            enable_cross_document: bool = True,
            output_format: str = "detailed",
            progress_callback: Optional[str] = None,
        ) -> Dict[str, Any]:
            _ = (
                batch_size,
                parallel_workers,
                enable_ocr,
                target_llm,
                chunk_strategy,
                enable_cross_document,
                output_format,
                progress_callback,
            )
            return {
                "status": "error",
                "pdf_sources": list(pdf_sources or []),
                "message": "PDF batch processing backend unavailable",
            }

        async def _cross_document_fallback(
            document_ids: Optional[List[str]] = None,
            analysis_types: Optional[List[str]] = None,
            similarity_threshold: float = 0.75,
            max_connections: int = 100,
            temporal_analysis: bool = True,
            include_visualizations: bool = False,
            output_format: str = "detailed",
        ) -> Dict[str, Any]:
            _ = analysis_types, similarity_threshold, max_connections, temporal_analysis, include_visualizations, output_format
            return {
                "status": "error",
                "document_ids": list(document_ids or []),
                "message": "PDF cross-document analysis backend unavailable",
            }

        async def _optimize_fallback(
            pdf_source: Union[str, Dict[str, Any], None] = None,
            target_llm: str = "gpt-4",
            chunk_strategy: str = "semantic",
            max_chunk_size: int = 4000,
            overlap_size: int = 200,
            preserve_structure: bool = True,
            include_metadata: bool = True,
        ) -> Dict[str, Any]:
            _ = target_llm, chunk_strategy, max_chunk_size, overlap_size, preserve_structure, include_metadata
            return {
                "status": "error",
                "pdf_source": pdf_source,
                "message": "PDF LLM optimization backend unavailable",
            }

        return {
            "pdf_analyze_relationships": _relationships_fallback,
            "pdf_query_corpus": _query_fallback,
            "pdf_extract_entities": _extract_fallback,
            "pdf_ingest_to_graphrag": _ingest_fallback,
            "pdf_batch_process": _batch_fallback,
            "pdf_cross_document_analysis": _cross_document_fallback,
            "pdf_optimize_for_llm": _optimize_fallback,
            "pdf_query_knowledge_graph": _knowledge_graph_fallback,
        }


_API = _load_pdf_tools_api()


def _error_result(message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return a deterministic error envelope for pdf-tools wrappers."""
    payload: Dict[str, Any] = {"status": "error", "error": message, "message": message}
    if context:
        payload.update(context)
    return payload


def _normalize_delegate_payload(payload: Any) -> Dict[str, Any]:
    """Normalize delegate payloads with deterministic error inference."""
    normalized = dict(payload or {})
    has_error = bool(normalized.get("error"))
    failed = normalized.get("success") is False or has_error
    if failed:
        normalized["status"] = "error"
    else:
        normalized.setdefault("status", "success")
    return normalized


async def _await_maybe(result: Any) -> Any:
    """Await coroutine-like values while preserving sync fallback behavior."""
    if hasattr(result, "__await__"):
        return await result
    return result


async def pdf_query_corpus(
    query: str,
    query_type: str = "hybrid",
    max_documents: int = 10,
    document_filters: Optional[Dict[str, Any]] = None,
    enable_reasoning: bool = True,
    include_sources: bool = True,
    confidence_threshold: float = 0.7,
) -> Dict[str, Any]:
    """Query ingested PDF corpus using semantic/hybrid retrieval."""
    normalized_query = str(query or "").strip()
    normalized_query_type = str(query_type or "").strip()

    if not normalized_query:
        return _error_result("query must be a non-empty string", {"query": query})
    if not normalized_query_type:
        return _error_result("query_type must be a non-empty string when provided")
    if not isinstance(max_documents, int) or max_documents < 1:
        return _error_result("max_documents must be an integer greater than or equal to 1")
    if document_filters is not None and not isinstance(document_filters, dict):
        return _error_result("document_filters must be an object when provided")
    if not isinstance(enable_reasoning, bool):
        return _error_result("enable_reasoning must be a boolean")
    if not isinstance(include_sources, bool):
        return _error_result("include_sources must be a boolean")
    if not isinstance(confidence_threshold, (int, float)):
        return _error_result("confidence_threshold must be a number")

    normalized_confidence = float(confidence_threshold)
    if normalized_confidence < 0.0 or normalized_confidence > 1.0:
        return _error_result("confidence_threshold must be between 0.0 and 1.0")

    try:
        payload = await _await_maybe(
            _API["pdf_query_corpus"](
                query=normalized_query,
                query_type=normalized_query_type,
                max_documents=max_documents,
                document_filters=document_filters,
                enable_reasoning=enable_reasoning,
                include_sources=include_sources,
                confidence_threshold=normalized_confidence,
            )
        )
    except Exception as exc:
        return _error_result(f"pdf_query_corpus failed: {exc}")

    normalized = _normalize_delegate_payload(payload)
    return normalized


async def pdf_analyze_relationships(
    document_id: str,
    analysis_type: str = "comprehensive",
    include_cross_document: bool = True,
    relationship_types: Optional[List[str]] = None,
    min_confidence: float = 0.6,
    max_relationships: int = 100,
) -> Dict[str, Any]:
    """Analyze intra-document and cross-document relationships in PDF content."""
    normalized_document_id = str(document_id or "").strip()
    normalized_analysis_type = str(analysis_type or "").strip()

    if not normalized_document_id:
        return _error_result("document_id must be a non-empty string", {"document_id": document_id})
    if not normalized_analysis_type:
        return _error_result("analysis_type must be a non-empty string when provided")
    if not isinstance(include_cross_document, bool):
        return _error_result("include_cross_document must be a boolean")
    if relationship_types is not None:
        if not isinstance(relationship_types, list) or any(not isinstance(item, str) or not item.strip() for item in relationship_types):
            return _error_result("relationship_types must be a list of non-empty strings when provided")
    if not isinstance(min_confidence, (int, float)):
        return _error_result("min_confidence must be a number")
    normalized_confidence = float(min_confidence)
    if normalized_confidence < 0.0 or normalized_confidence > 1.0:
        return _error_result("min_confidence must be between 0.0 and 1.0")
    if not isinstance(max_relationships, int) or max_relationships < 1:
        return _error_result("max_relationships must be an integer greater than or equal to 1")

    try:
        payload = await _await_maybe(
            _API["pdf_analyze_relationships"](
                document_id=normalized_document_id,
                analysis_type=normalized_analysis_type,
                include_cross_document=include_cross_document,
                relationship_types=relationship_types,
                min_confidence=normalized_confidence,
                max_relationships=max_relationships,
            )
        )
    except Exception as exc:
        return _error_result(f"pdf_analyze_relationships failed: {exc}")

    normalized = _normalize_delegate_payload(payload)
    return normalized


async def pdf_extract_entities(
    pdf_source: Union[str, dict, None] = None,
    entity_types: Optional[List[str]] = None,
    extraction_method: str = "hybrid",
    confidence_threshold: float = 0.7,
    include_relationships: bool = True,
    context_window: int = 3,
    custom_patterns: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Extract entities and relationships from PDF content."""
    normalized_method = str(extraction_method or "").strip()

    if pdf_source is None:
        return _error_result("pdf_source must be provided")
    if isinstance(pdf_source, str) and not pdf_source.strip():
        return _error_result("pdf_source must be a non-empty string when provided as a string")
    if not isinstance(pdf_source, (str, dict)):
        return _error_result("pdf_source must be a string or object")
    if entity_types is not None:
        if not isinstance(entity_types, list) or any(not isinstance(item, str) or not item.strip() for item in entity_types):
            return _error_result("entity_types must be a list of non-empty strings when provided")
    if not normalized_method:
        return _error_result("extraction_method must be a non-empty string when provided")
    if not isinstance(confidence_threshold, (int, float)):
        return _error_result("confidence_threshold must be a number")
    normalized_confidence = float(confidence_threshold)
    if normalized_confidence < 0.0 or normalized_confidence > 1.0:
        return _error_result("confidence_threshold must be between 0.0 and 1.0")
    if not isinstance(include_relationships, bool):
        return _error_result("include_relationships must be a boolean")
    if not isinstance(context_window, int) or context_window < 0:
        return _error_result("context_window must be an integer greater than or equal to 0")
    if custom_patterns is not None and not isinstance(custom_patterns, dict):
        return _error_result("custom_patterns must be an object when provided")

    try:
        payload = await _await_maybe(
            _API["pdf_extract_entities"](
                pdf_source=pdf_source,
                entity_types=entity_types,
                extraction_method=normalized_method,
                confidence_threshold=normalized_confidence,
                include_relationships=include_relationships,
                context_window=context_window,
                custom_patterns=custom_patterns,
            )
        )
    except Exception as exc:
        return _error_result(f"pdf_extract_entities failed: {exc}")

    normalized = _normalize_delegate_payload(payload)
    return normalized


async def pdf_batch_process(
    pdf_sources: Optional[List[Union[str, Dict[str, Any]]]] = None,
    batch_size: int = 5,
    parallel_workers: int = 3,
    enable_ocr: bool = True,
    target_llm: str = "gpt-4",
    chunk_strategy: str = "semantic",
    enable_cross_document: bool = True,
    output_format: str = "detailed",
    progress_callback: Optional[str] = None,
) -> Dict[str, Any]:
    """Batch-process one or more PDFs through ingestion/extraction pipelines."""
    if not isinstance(pdf_sources, list) or not pdf_sources:
        return _error_result("pdf_sources must be provided as a non-empty array", {"pdf_sources": list(pdf_sources or [])})
    if any(
        not isinstance(source, (str, dict))
        or (isinstance(source, str) and not source.strip())
        for source in pdf_sources
    ):
        return _error_result("pdf_sources entries must be non-empty strings or objects")
    if not isinstance(batch_size, int) or batch_size < 1:
        return _error_result("batch_size must be an integer greater than or equal to 1")
    if not isinstance(parallel_workers, int) or parallel_workers < 1:
        return _error_result("parallel_workers must be an integer greater than or equal to 1")
    if not isinstance(enable_ocr, bool):
        return _error_result("enable_ocr must be a boolean")
    if not isinstance(enable_cross_document, bool):
        return _error_result("enable_cross_document must be a boolean")
    if not isinstance(target_llm, str) or not target_llm.strip():
        return _error_result("target_llm must be a non-empty string")
    if not isinstance(chunk_strategy, str) or not chunk_strategy.strip():
        return _error_result("chunk_strategy must be a non-empty string")
    if not isinstance(output_format, str) or not output_format.strip():
        return _error_result("output_format must be a non-empty string")
    if progress_callback is not None and (not isinstance(progress_callback, str) or not progress_callback.strip()):
        return _error_result("progress_callback must be a non-empty string when provided")

    try:
        payload = await _await_maybe(
            _API["pdf_batch_process"](
                pdf_sources=pdf_sources,
                batch_size=batch_size,
                parallel_workers=parallel_workers,
                enable_ocr=enable_ocr,
                target_llm=target_llm.strip(),
                chunk_strategy=chunk_strategy.strip(),
                enable_cross_document=enable_cross_document,
                output_format=output_format.strip(),
                progress_callback=progress_callback.strip() if isinstance(progress_callback, str) else None,
            )
        )
    except Exception as exc:
        return _error_result(f"pdf_batch_process failed: {exc}")

    normalized = _normalize_delegate_payload(payload)
    return normalized


async def pdf_ingest_to_graphrag(
    pdf_source: Union[str, Dict[str, Any], None] = None,
    collection_name: str = "default",
    metadata: Optional[Dict[str, Any]] = None,
    chunk_strategy: str = "semantic",
    max_chunk_size: int = 4000,
    overlap_size: int = 200,
    extract_entities: bool = True,
    build_knowledge_graph: bool = True,
    store_embeddings: bool = True,
) -> Dict[str, Any]:
    """Ingest a PDF into the GraphRAG pipeline with normalized validation."""
    if pdf_source is None:
        return _error_result("pdf_source must be provided")
    if isinstance(pdf_source, str) and not pdf_source.strip():
        return _error_result("pdf_source must be a non-empty string when provided as a string")
    if not isinstance(pdf_source, (str, dict)):
        return _error_result("pdf_source must be a string or object")
    if isinstance(pdf_source, dict) and not any(key in pdf_source for key in ("path", "url", "cid", "content")):
        return _error_result("pdf_source object must include at least one of: path, url, cid, content")
    if not isinstance(collection_name, str) or not collection_name.strip():
        return _error_result("collection_name must be a non-empty string")
    if metadata is not None and not isinstance(metadata, dict):
        return _error_result("metadata must be an object when provided")
    if not isinstance(chunk_strategy, str) or not chunk_strategy.strip():
        return _error_result("chunk_strategy must be a non-empty string")
    if not isinstance(max_chunk_size, int) or max_chunk_size < 1:
        return _error_result("max_chunk_size must be an integer greater than or equal to 1")
    if not isinstance(overlap_size, int) or overlap_size < 0:
        return _error_result("overlap_size must be an integer greater than or equal to 0")
    if overlap_size > max_chunk_size:
        return _error_result("overlap_size must be less than or equal to max_chunk_size")
    if not isinstance(extract_entities, bool):
        return _error_result("extract_entities must be a boolean")
    if not isinstance(build_knowledge_graph, bool):
        return _error_result("build_knowledge_graph must be a boolean")
    if not isinstance(store_embeddings, bool):
        return _error_result("store_embeddings must be a boolean")

    try:
        payload = await _await_maybe(
            _API["pdf_ingest_to_graphrag"](
                pdf_source=pdf_source,
                collection_name=collection_name.strip(),
                metadata=metadata,
                chunk_strategy=chunk_strategy.strip(),
                max_chunk_size=max_chunk_size,
                overlap_size=overlap_size,
                extract_entities=extract_entities,
                build_knowledge_graph=build_knowledge_graph,
                store_embeddings=store_embeddings,
            )
        )
    except Exception as exc:
        return _error_result(f"pdf_ingest_to_graphrag failed: {exc}")

    normalized = _normalize_delegate_payload(payload)
    return normalized


async def pdf_cross_document_analysis(
    document_ids: Optional[List[str]] = None,
    analysis_types: Optional[List[str]] = None,
    similarity_threshold: float = 0.75,
    max_connections: int = 100,
    temporal_analysis: bool = True,
    include_visualizations: bool = False,
    output_format: str = "detailed",
) -> Dict[str, Any]:
    """Analyze relationships and themes across multiple PDFs."""
    if not isinstance(document_ids, list) or not document_ids:
        return _error_result(
            "document_ids must be provided as a non-empty array",
            {"document_ids": list(document_ids or [])},
        )
    if any(not isinstance(item, str) or not item.strip() for item in document_ids):
        return _error_result("document_ids entries must be non-empty strings")
    if analysis_types is not None:
        if not isinstance(analysis_types, list) or any(not isinstance(item, str) or not item.strip() for item in analysis_types):
            return _error_result("analysis_types must be a list of non-empty strings when provided")
    if not isinstance(similarity_threshold, (int, float)):
        return _error_result("similarity_threshold must be a number")
    normalized_similarity = float(similarity_threshold)
    if normalized_similarity < 0.0 or normalized_similarity > 1.0:
        return _error_result("similarity_threshold must be between 0.0 and 1.0")
    if not isinstance(max_connections, int) or max_connections < 1:
        return _error_result("max_connections must be an integer greater than or equal to 1")
    if not isinstance(temporal_analysis, bool):
        return _error_result("temporal_analysis must be a boolean")
    if not isinstance(include_visualizations, bool):
        return _error_result("include_visualizations must be a boolean")
    if not isinstance(output_format, str) or not output_format.strip():
        return _error_result("output_format must be a non-empty string")

    try:
        payload = await _await_maybe(
            _API["pdf_cross_document_analysis"](
                document_ids=document_ids,
                analysis_types=analysis_types,
                similarity_threshold=normalized_similarity,
                max_connections=max_connections,
                temporal_analysis=temporal_analysis,
                include_visualizations=include_visualizations,
                output_format=output_format.strip(),
            )
        )
    except Exception as exc:
        return _error_result(f"pdf_cross_document_analysis failed: {exc}")

    normalized = _normalize_delegate_payload(payload)
    return normalized


async def pdf_optimize_for_llm(
    pdf_source: Union[str, Dict[str, Any], None] = None,
    target_llm: str = "gpt-4",
    chunk_strategy: str = "semantic",
    max_chunk_size: int = 4000,
    overlap_size: int = 200,
    preserve_structure: bool = True,
    include_metadata: bool = True,
) -> Dict[str, Any]:
    """Optimize PDF text/chunks for downstream LLM use."""
    if pdf_source is None:
        return _error_result("pdf_source must be provided")
    if isinstance(pdf_source, str) and not pdf_source.strip():
        return _error_result("pdf_source must be a non-empty string when provided as a string")
    if not isinstance(pdf_source, (str, dict)):
        return _error_result("pdf_source must be a string or object")
    if not isinstance(target_llm, str) or not target_llm.strip():
        return _error_result("target_llm must be a non-empty string")
    if not isinstance(chunk_strategy, str) or not chunk_strategy.strip():
        return _error_result("chunk_strategy must be a non-empty string")
    if not isinstance(max_chunk_size, int) or max_chunk_size < 1:
        return _error_result("max_chunk_size must be an integer greater than or equal to 1")
    if not isinstance(overlap_size, int) or overlap_size < 0:
        return _error_result("overlap_size must be an integer greater than or equal to 0")
    if overlap_size > max_chunk_size:
        return _error_result("overlap_size must be less than or equal to max_chunk_size")
    if not isinstance(preserve_structure, bool):
        return _error_result("preserve_structure must be a boolean")
    if not isinstance(include_metadata, bool):
        return _error_result("include_metadata must be a boolean")

    try:
        payload = await _await_maybe(
            _API["pdf_optimize_for_llm"](
                pdf_source=pdf_source,
                target_llm=target_llm.strip(),
                chunk_strategy=chunk_strategy.strip(),
                max_chunk_size=max_chunk_size,
                overlap_size=overlap_size,
                preserve_structure=preserve_structure,
                include_metadata=include_metadata,
            )
        )
    except Exception as exc:
        return _error_result(f"pdf_optimize_for_llm failed: {exc}")

    normalized = _normalize_delegate_payload(payload)
    return normalized


async def pdf_query_knowledge_graph(
    graph_id: str,
    query: str,
    query_type: str = "sparql",
    max_results: int = 100,
    include_metadata: bool = True,
    return_subgraph: bool = False,
) -> Dict[str, Any]:
    """Query a knowledge graph generated from processed PDFs."""
    normalized_graph_id = str(graph_id or "").strip()
    normalized_query = str(query or "").strip()
    normalized_query_type = str(query_type or "").strip()
    valid_query_types = {"sparql", "cypher", "entity", "relationship", "natural_language"}

    if not normalized_graph_id:
        return _error_result("graph_id must be a non-empty string", {"graph_id": graph_id})
    if not normalized_query:
        return _error_result("query must be a non-empty string", {"query": query})
    if normalized_query_type not in valid_query_types:
        return _error_result(
            f"query_type must be one of: {sorted(valid_query_types)}",
            {"query_type": query_type},
        )
    if not isinstance(max_results, int) or max_results < 1:
        return _error_result("max_results must be an integer greater than or equal to 1")
    if not isinstance(include_metadata, bool):
        return _error_result("include_metadata must be a boolean")
    if not isinstance(return_subgraph, bool):
        return _error_result("return_subgraph must be a boolean")

    try:
        payload = await _await_maybe(
            _API["pdf_query_knowledge_graph"](
                graph_id=normalized_graph_id,
                query=normalized_query,
                query_type=normalized_query_type,
                max_results=max_results,
                include_metadata=include_metadata,
                return_subgraph=return_subgraph,
            )
        )
    except Exception as exc:
        return _error_result(f"pdf_query_knowledge_graph failed: {exc}")

    normalized = _normalize_delegate_payload(payload)
    return normalized


def register_native_pdf_tools(manager: Any) -> None:
    """Register native pdf-tools category tools in unified manager."""
    manager.register_tool(
        category="pdf_tools",
        name="pdf_ingest_to_graphrag",
        func=pdf_ingest_to_graphrag,
        description="Ingest a PDF into the GraphRAG processing pipeline.",
        input_schema={
            "type": "object",
            "properties": {
                "pdf_source": {
                    "type": ["string", "object", "null"],
                    "properties": {
                        "path": {"type": "string", "minLength": 1},
                        "url": {"type": "string", "minLength": 1},
                        "cid": {"type": "string", "minLength": 1},
                        "content": {"type": "string", "minLength": 1}
                    },
                    "additionalProperties": True
                },
                "collection_name": {"type": "string", "minLength": 1, "default": "default"},
                "metadata": {"type": ["object", "null"]},
                "chunk_strategy": {"type": "string", "minLength": 1, "default": "semantic"},
                "max_chunk_size": {"type": "integer", "minimum": 1, "default": 4000},
                "overlap_size": {"type": "integer", "minimum": 0, "default": 200},
                "extract_entities": {"type": "boolean", "default": True},
                "build_knowledge_graph": {"type": "boolean", "default": True},
                "store_embeddings": {"type": "boolean", "default": True}
            },
            "required": []
        },
        runtime="fastapi",
        tags=["native", "mcpp", "pdf-tools"],
    )
    manager.register_tool(
        category="pdf_tools",
        name="pdf_analyze_relationships",
        func=pdf_analyze_relationships,
        description="Analyze entity relationships in processed PDFs.",
        input_schema={
            "type": "object",
            "properties": {
                "document_id": {"type": "string", "minLength": 1},
                "analysis_type": {"type": "string", "minLength": 1, "default": "comprehensive"},
                "include_cross_document": {"type": "boolean", "default": True},
                "relationship_types": {"type": ["array", "null"], "items": {"type": "string"}},
                "min_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.6},
                "max_relationships": {"type": "integer", "minimum": 1, "default": 100},
            },
            "required": ["document_id"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "pdf-tools"],
    )

    manager.register_tool(
        category="pdf_tools",
        name="pdf_query_corpus",
        func=pdf_query_corpus,
        description="Query a processed PDF corpus.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "query_type": {"type": "string", "minLength": 1},
                "max_documents": {"type": "integer", "minimum": 1},
                "document_filters": {"type": ["object", "null"]},
                "enable_reasoning": {"type": "boolean"},
                "include_sources": {"type": "boolean"},
                "confidence_threshold": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            },
            "required": ["query"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "pdf-tools"],
    )

    manager.register_tool(
        category="pdf_tools",
        name="pdf_extract_entities",
        func=pdf_extract_entities,
        description="Extract entities and links from PDF content.",
        input_schema={
            "type": "object",
            "properties": {
                "pdf_source": {"type": ["string", "object", "null"]},
                "entity_types": {"type": ["array", "null"], "items": {"type": "string"}},
                "extraction_method": {"type": "string", "minLength": 1},
                "confidence_threshold": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "include_relationships": {"type": "boolean"},
                "context_window": {"type": "integer", "minimum": 0},
                "custom_patterns": {"type": ["object", "null"]},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "pdf-tools"],
    )

    manager.register_tool(
        category="pdf_tools",
        name="pdf_batch_process",
        func=pdf_batch_process,
        description="Batch process multiple PDF sources for extraction and analysis.",
        input_schema={
            "type": "object",
            "properties": {
                "pdf_sources": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": ["string", "object"],
                    },
                },
                "batch_size": {"type": "integer", "minimum": 1, "default": 5},
                "parallel_workers": {"type": "integer", "minimum": 1, "default": 3},
                "enable_ocr": {"type": "boolean", "default": True},
                "target_llm": {"type": "string", "minLength": 1, "default": "gpt-4"},
                "chunk_strategy": {"type": "string", "minLength": 1, "default": "semantic"},
                "enable_cross_document": {"type": "boolean", "default": True},
                "output_format": {"type": "string", "minLength": 1, "default": "detailed"},
                "progress_callback": {"type": ["string", "null"]},
            },
            "required": ["pdf_sources"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "pdf-tools"],
    )

    manager.register_tool(
        category="pdf_tools",
        name="pdf_cross_document_analysis",
        func=pdf_cross_document_analysis,
        description="Analyze entities, themes, and citations across multiple PDFs.",
        input_schema={
            "type": "object",
            "properties": {
                "document_ids": {"type": "array", "minItems": 1, "items": {"type": "string"}},
                "analysis_types": {"type": ["array", "null"], "items": {"type": "string"}},
                "similarity_threshold": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.75},
                "max_connections": {"type": "integer", "minimum": 1, "default": 100},
                "temporal_analysis": {"type": "boolean", "default": True},
                "include_visualizations": {"type": "boolean", "default": False},
                "output_format": {"type": "string", "minLength": 1, "default": "detailed"},
            },
            "required": ["document_ids"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "pdf-tools"],
    )

    manager.register_tool(
        category="pdf_tools",
        name="pdf_optimize_for_llm",
        func=pdf_optimize_for_llm,
        description="Optimize PDF content for LLM chunking and downstream use.",
        input_schema={
            "type": "object",
            "properties": {
                "pdf_source": {"type": ["string", "object", "null"]},
                "target_llm": {"type": "string", "minLength": 1, "default": "gpt-4"},
                "chunk_strategy": {"type": "string", "minLength": 1, "default": "semantic"},
                "max_chunk_size": {"type": "integer", "minimum": 1, "default": 4000},
                "overlap_size": {"type": "integer", "minimum": 0, "default": 200},
                "preserve_structure": {"type": "boolean", "default": True},
                "include_metadata": {"type": "boolean", "default": True},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "pdf-tools"],
    )

    manager.register_tool(
        category="pdf_tools",
        name="pdf_query_knowledge_graph",
        func=pdf_query_knowledge_graph,
        description="Query a PDF-derived knowledge graph.",
        input_schema={
            "type": "object",
            "properties": {
                "graph_id": {"type": "string", "minLength": 1},
                "query": {"type": "string", "minLength": 1},
                "query_type": {
                    "type": "string",
                    "enum": ["sparql", "cypher", "entity", "relationship", "natural_language"],
                    "default": "sparql",
                },
                "max_results": {"type": "integer", "minimum": 1, "default": 100},
                "include_metadata": {"type": "boolean", "default": True},
                "return_subgraph": {"type": "boolean", "default": False},
            },
            "required": ["graph_id", "query"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "pdf-tools"],
    )
