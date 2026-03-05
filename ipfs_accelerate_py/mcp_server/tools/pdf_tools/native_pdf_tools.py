"""Native pdf-tools category implementations for unified mcp_server."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def _load_pdf_tools_api() -> Dict[str, Any]:
    """Resolve source pdf-tools APIs with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.pdf_tools import (  # type: ignore
            pdf_batch_process as _pdf_batch_process,
            pdf_extract_entities as _pdf_extract_entities,
            pdf_query_corpus as _pdf_query_corpus,
        )

        return {
            "pdf_query_corpus": _pdf_query_corpus,
            "pdf_extract_entities": _pdf_extract_entities,
            "pdf_batch_process": _pdf_batch_process,
        }
    except Exception:
        logger.warning("Source pdf_tools import unavailable, using fallback pdf-tools functions")

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

        return {
            "pdf_query_corpus": _query_fallback,
            "pdf_extract_entities": _extract_fallback,
            "pdf_batch_process": _batch_fallback,
        }


_API = _load_pdf_tools_api()


def _error_result(message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return a deterministic error envelope for pdf-tools wrappers."""
    payload: Dict[str, Any] = {"status": "error", "error": message, "message": message}
    if context:
        payload.update(context)
    return payload


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

    normalized = dict(payload or {})
    normalized.setdefault("status", "error" if "error" in normalized else "success")
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

    normalized = dict(payload or {})
    normalized.setdefault("status", "error" if "error" in normalized else "success")
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

    normalized = dict(payload or {})
    normalized.setdefault("status", "error" if "error" in normalized else "success")
    return normalized


def register_native_pdf_tools(manager: Any) -> None:
    """Register native pdf-tools category tools in unified manager."""
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
