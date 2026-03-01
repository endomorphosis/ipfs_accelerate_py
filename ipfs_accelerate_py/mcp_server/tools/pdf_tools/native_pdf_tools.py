"""Native pdf-tools category implementations for unified mcp_server."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def _load_pdf_tools_api() -> Dict[str, Any]:
    """Resolve source pdf-tools APIs with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.pdf_tools import (  # type: ignore
            pdf_extract_entities as _pdf_extract_entities,
            pdf_query_corpus as _pdf_query_corpus,
        )

        return {
            "pdf_query_corpus": _pdf_query_corpus,
            "pdf_extract_entities": _pdf_extract_entities,
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

        return {
            "pdf_query_corpus": _query_fallback,
            "pdf_extract_entities": _extract_fallback,
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
