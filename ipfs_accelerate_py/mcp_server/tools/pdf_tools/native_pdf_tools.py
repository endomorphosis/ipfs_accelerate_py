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
    result = _API["pdf_query_corpus"](
        query=query,
        query_type=query_type,
        max_documents=max_documents,
        document_filters=document_filters,
        enable_reasoning=enable_reasoning,
        include_sources=include_sources,
        confidence_threshold=confidence_threshold,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


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
    result = _API["pdf_extract_entities"](
        pdf_source=pdf_source,
        entity_types=entity_types,
        extraction_method=extraction_method,
        confidence_threshold=confidence_threshold,
        include_relationships=include_relationships,
        context_window=context_window,
        custom_patterns=custom_patterns,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


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
    if not pdf_sources:
        return {
            "status": "error",
            "message": "pdf_sources must be provided",
            "pdf_sources": [],
        }

    result = _API["pdf_batch_process"](
        pdf_sources=pdf_sources,
        batch_size=int(batch_size),
        parallel_workers=int(parallel_workers),
        enable_ocr=bool(enable_ocr),
        target_llm=target_llm,
        chunk_strategy=chunk_strategy,
        enable_cross_document=bool(enable_cross_document),
        output_format=output_format,
        progress_callback=progress_callback,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


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
                "query_type": {"type": "string"},
                "max_documents": {"type": "integer"},
                "document_filters": {"type": ["object", "null"]},
                "enable_reasoning": {"type": "boolean"},
                "include_sources": {"type": "boolean"},
                "confidence_threshold": {"type": "number"},
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
                "extraction_method": {"type": "string"},
                "confidence_threshold": {"type": "number"},
                "include_relationships": {"type": "boolean"},
                "context_window": {"type": "integer"},
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
                    "items": {
                        "type": ["string", "object"],
                    },
                },
                "batch_size": {"type": "integer", "default": 5},
                "parallel_workers": {"type": "integer", "default": 3},
                "enable_ocr": {"type": "boolean", "default": True},
                "target_llm": {"type": "string", "default": "gpt-4"},
                "chunk_strategy": {"type": "string", "default": "semantic"},
                "enable_cross_document": {"type": "boolean", "default": True},
                "output_format": {"type": "string", "default": "detailed"},
                "progress_callback": {"type": ["string", "null"]},
            },
            "required": ["pdf_sources"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "pdf-tools"],
    )
