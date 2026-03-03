"""Native dataset tool implementations for unified mcp_server."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _load_dataset_api() -> Dict[str, Any]:
    """Resolve source dataset APIs with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.dataset_tools import (  # type: ignore
            convert_dataset_format as _convert_dataset_format,
            legal_text_to_deontic as _legal_text_to_deontic,
            load_dataset as _load_dataset,
            process_dataset as _process_dataset,
            save_dataset as _save_dataset,
            text_to_fol as _text_to_fol,
        )

        return {
            "load_dataset": _load_dataset,
            "save_dataset": _save_dataset,
            "process_dataset": _process_dataset,
            "convert_dataset_format": _convert_dataset_format,
            "text_to_fol": _text_to_fol,
            "legal_text_to_deontic": _legal_text_to_deontic,
        }
    except Exception:
        logger.warning("Source dataset_tools import unavailable, using fallback dataset functions")

        async def _load_fallback(
            source: str,
            format: Optional[str] = None,
            options: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            _ = format, options
            return {
                "status": "success",
                "source": source,
                "dataset_id": "fallback-dataset",
                "message": "Fallback dataset loader used",
            }

        async def _save_fallback(
            dataset_data: str | Dict[str, Any],
            destination: Optional[str] = None,
            format: Optional[str] = None,
            options: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            _ = dataset_data, format, options
            if not destination:
                return {
                    "status": "error",
                    "message": "Destination must be provided",
                }
            return {
                "status": "success",
                "dataset_id": "fallback-dataset",
                "destination": destination,
                "format": format or "json",
            }

        async def _process_fallback(
            dataset_source: str | Dict[str, Any] | Any,
            operations: Optional[List[Dict[str, Any]]] = None,
            output_id: Optional[str] = None,
        ) -> Dict[str, Any]:
            _ = operations
            return {
                "status": "success",
                "dataset_id": output_id or "processed-fallback-dataset",
                "original_dataset_id": str(dataset_source)[:50],
                "num_operations": len(operations or []),
                "num_records": 0,
            }

        async def _convert_fallback(
            dataset_id: str,
            target_format: Optional[str] = None,
            output_path: Optional[str] = None,
            options: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            _ = output_path, options
            if not target_format:
                return {
                    "status": "error",
                    "message": "target_format must be provided",
                    "dataset_id": dataset_id,
                }
            return {
                "status": "success",
                "original_dataset_id": dataset_id,
                "dataset_id": f"converted_{dataset_id}_{target_format}",
                "target_format": target_format,
            }

        async def _text_to_fol_fallback(
            text_input: str | Dict[str, Any],
            domain_predicates: Optional[List[str]] = None,
            output_format: str = "json",
            include_metadata: bool = True,
            confidence_threshold: float = 0.5,
        ) -> Dict[str, Any]:
            normalized_text = str(text_input if not isinstance(text_input, dict) else text_input.get("text", "")).strip()
            if not normalized_text:
                return {
                    "status": "error",
                    "message": "text_input must be provided",
                    "fol": None,
                }
            return {
                "status": "success",
                "output_format": output_format,
                "fol": f"TEXT({normalized_text[:80]})",
                "domain_predicates": list(domain_predicates or []),
                "metadata": {
                    "include_metadata": bool(include_metadata),
                    "confidence_threshold": float(confidence_threshold),
                },
            }

        async def _legal_text_to_deontic_fallback(
            text_input: str | Dict[str, Any],
            jurisdiction: str = "us",
            document_type: str = "statute",
            output_format: str = "json",
            extract_obligations: bool = True,
            include_exceptions: bool = True,
        ) -> Dict[str, Any]:
            normalized_text = str(text_input if not isinstance(text_input, dict) else text_input.get("text", "")).strip()
            if not normalized_text:
                return {
                    "status": "error",
                    "message": "text_input must be provided",
                    "deontic": None,
                }
            return {
                "status": "success",
                "output_format": output_format,
                "deontic": f"OBLIGATION({normalized_text[:80]})",
                "jurisdiction": str(jurisdiction or "us"),
                "document_type": str(document_type or "statute"),
                "extract_obligations": bool(extract_obligations),
                "include_exceptions": bool(include_exceptions),
            }

        return {
            "load_dataset": _load_fallback,
            "save_dataset": _save_fallback,
            "process_dataset": _process_fallback,
            "convert_dataset_format": _convert_fallback,
            "text_to_fol": _text_to_fol_fallback,
            "legal_text_to_deontic": _legal_text_to_deontic_fallback,
        }


_API = _load_dataset_api()


async def load_dataset(
    source: str,
    format: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load a dataset from a source identifier."""
    result = _API["load_dataset"](source=source, format=format, options=options)
    if hasattr(result, "__await__"):
        return await result
    return result


async def save_dataset(
    dataset_data: str | Dict[str, Any],
    destination: Optional[str] = None,
    format: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Save a dataset to a target destination and format."""
    result = _API["save_dataset"](
        dataset_data=dataset_data,
        destination=destination,
        format=format,
        options=options,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def process_dataset(
    dataset_source: str | Dict[str, Any] | Any,
    operations: Optional[List[Dict[str, Any]]] = None,
    output_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Process a dataset using a list of transformation operations."""
    result = _API["process_dataset"](
        dataset_source=dataset_source,
        operations=operations,
        output_id=output_id,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def convert_dataset_format(
    dataset_id: str,
    target_format: Optional[str] = None,
    output_path: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Convert an existing dataset into a target format."""
    result = _API["convert_dataset_format"](
        dataset_id=dataset_id,
        target_format=target_format,
        output_path=output_path,
        options=options,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def text_to_fol(
    text_input: str | Dict[str, Any],
    domain_predicates: Optional[List[str]] = None,
    output_format: str = "json",
    include_metadata: bool = True,
    confidence_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Convert natural-language text into first-order logic payloads."""
    result = _API["text_to_fol"](
        text_input=text_input,
        domain_predicates=domain_predicates,
        output_format=output_format,
        include_metadata=include_metadata,
        confidence_threshold=confidence_threshold,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def legal_text_to_deontic(
    text_input: str | Dict[str, Any],
    jurisdiction: str = "us",
    document_type: str = "statute",
    output_format: str = "json",
    extract_obligations: bool = True,
    include_exceptions: bool = True,
) -> Dict[str, Any]:
    """Convert legal text into deontic-logic payloads."""
    result = _API["legal_text_to_deontic"](
        text_input=text_input,
        jurisdiction=jurisdiction,
        document_type=document_type,
        output_format=output_format,
        extract_obligations=extract_obligations,
        include_exceptions=include_exceptions,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


def register_native_dataset_tools(manager: Any) -> None:
    """Register native dataset tools in unified hierarchical manager."""
    manager.register_tool(
        category="dataset_tools",
        name="load_dataset",
        func=load_dataset,
        description="Load a dataset from a source identifier.",
        input_schema={
            "type": "object",
            "properties": {
                "source": {"type": "string"},
                "format": {"type": ["string", "null"]},
                "options": {"type": ["object", "null"]},
            },
            "required": ["source"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "dataset"],
    )

    manager.register_tool(
        category="dataset_tools",
        name="save_dataset",
        func=save_dataset,
        description="Save dataset content or a dataset ID to a destination.",
        input_schema={
            "type": "object",
            "properties": {
                "dataset_data": {"type": ["string", "object"]},
                "destination": {"type": ["string", "null"]},
                "format": {"type": ["string", "null"]},
                "options": {"type": ["object", "null"]},
            },
            "required": ["dataset_data"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "dataset"],
    )

    manager.register_tool(
        category="dataset_tools",
        name="process_dataset",
        func=process_dataset,
        description="Apply dataset processing operations and produce a transformed dataset.",
        input_schema={
            "type": "object",
            "properties": {
                "dataset_source": {"type": ["string", "object"]},
                "operations": {
                    "type": ["array", "null"],
                    "items": {"type": "object"},
                },
                "output_id": {"type": ["string", "null"]},
            },
            "required": ["dataset_source"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "dataset"],
    )

    manager.register_tool(
        category="dataset_tools",
        name="convert_dataset_format",
        func=convert_dataset_format,
        description="Convert a dataset to a target serialization format.",
        input_schema={
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string"},
                "target_format": {"type": ["string", "null"]},
                "output_path": {"type": ["string", "null"]},
                "options": {"type": ["object", "null"]},
            },
            "required": ["dataset_id"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "dataset"],
    )

    manager.register_tool(
        category="dataset_tools",
        name="text_to_fol",
        func=text_to_fol,
        description="Convert natural language text into first-order logic expressions.",
        input_schema={
            "type": "object",
            "properties": {
                "text_input": {"type": ["string", "object"]},
                "domain_predicates": {"type": ["array", "null"], "items": {"type": "string"}},
                "output_format": {"type": "string", "default": "json"},
                "include_metadata": {"type": "boolean", "default": True},
                "confidence_threshold": {"type": "number", "default": 0.5},
            },
            "required": ["text_input"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "dataset", "logic"],
    )

    manager.register_tool(
        category="dataset_tools",
        name="legal_text_to_deontic",
        func=legal_text_to_deontic,
        description="Convert legal text into deontic logic expressions.",
        input_schema={
            "type": "object",
            "properties": {
                "text_input": {"type": ["string", "object"]},
                "jurisdiction": {"type": "string", "default": "us"},
                "document_type": {"type": "string", "default": "statute"},
                "output_format": {"type": "string", "default": "json"},
                "extract_obligations": {"type": "boolean", "default": True},
                "include_exceptions": {"type": "boolean", "default": True},
            },
            "required": ["text_input"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "dataset", "logic"],
    )
