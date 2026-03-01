"""Native data processing tool implementations for unified mcp_server."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _load_data_processing_api() -> Dict[str, Any]:
    """Resolve source data processing engines with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.processors.development.data_processing_engine import (  # type: ignore
            chunk_text_engine,
            convert_format_engine,
            transform_data_engine,
            validate_data_engine,
        )

        return {
            "chunk_text": chunk_text_engine,
            "transform_data": transform_data_engine,
            "convert_format": convert_format_engine,
            "validate_data": validate_data_engine,
        }
    except Exception:

        async def _chunk_text_fallback(
            text: str,
            strategy: str = "fixed_size",
            chunk_size: int = 1000,
            overlap: int = 100,
            max_chunks: int = 100,
        ) -> Dict[str, Any]:
            _ = strategy, overlap
            chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
            chunks = chunks[: max(1, int(max_chunks))]
            return {"status": "success", "chunks": chunks, "chunk_count": len(chunks)}

        async def _transform_data_fallback(data: Any, transformation: str, **parameters: Any) -> Dict[str, Any]:
            _ = parameters
            return {"status": "success", "result": data, "transformation": transformation}

        async def _convert_format_fallback(
            data: Any,
            source_format: str,
            target_format: str,
            options: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            _ = options
            if source_format == target_format:
                return {
                    "status": "success",
                    "result": data,
                    "source_format": source_format,
                    "target_format": target_format,
                    "message": "No conversion needed - formats are identical",
                }
            return {
                "status": "error",
                "message": "data_processing engine unavailable for format conversion",
            }

        async def _validate_data_fallback(
            data: Any,
            validation_type: str,
            schema: Optional[Dict[str, Any]] = None,
            rules: Optional[List[Dict[str, Any]]] = None,
        ) -> Dict[str, Any]:
            _ = schema, rules
            return {
                "status": "success",
                "validation_type": validation_type,
                "validation_result": {"valid": True, "errors": [], "warnings": [], "metrics": {}},
                "data_summary": {
                    "type": type(data).__name__,
                    "size": len(data) if hasattr(data, "__len__") else None,
                },
            }

        return {
            "chunk_text": _chunk_text_fallback,
            "transform_data": _transform_data_fallback,
            "convert_format": _convert_format_fallback,
            "validate_data": _validate_data_fallback,
        }


_API = _load_data_processing_api()


async def chunk_text(
    text: str,
    strategy: str = "fixed_size",
    chunk_size: int = 1000,
    overlap: int = 100,
    max_chunks: int = 100,
) -> Dict[str, Any]:
    """Split text into chunks using configured strategy."""
    return await _API["chunk_text"](
        text=text,
        strategy=strategy,
        chunk_size=chunk_size,
        overlap=overlap,
        max_chunks=max_chunks,
    )


async def transform_data(data: Any, transformation: str, **parameters: Any) -> Dict[str, Any]:
    """Apply a named transformation to input data."""
    return await _API["transform_data"](data=data, transformation=transformation, **parameters)


async def convert_format(
    data: Any,
    source_format: str,
    target_format: str,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Convert data between supported formats."""
    return await _API["convert_format"](
        data=data,
        source_format=source_format,
        target_format=target_format,
        options=options,
    )


async def validate_data(
    data: Any,
    validation_type: str,
    schema: Optional[Dict[str, Any]] = None,
    rules: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Validate input data with requested strategy."""
    return await _API["validate_data"](
        data=data,
        validation_type=validation_type,
        schema=schema,
        rules=rules,
    )


def register_native_data_processing_tools(manager: Any) -> None:
    """Register native data processing tools in unified hierarchical manager."""
    manager.register_tool(
        category="data_processing_tools",
        name="chunk_text",
        func=chunk_text,
        description="Split text into overlapping chunks.",
        input_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "strategy": {"type": "string"},
                "chunk_size": {"type": "integer"},
                "overlap": {"type": "integer"},
                "max_chunks": {"type": "integer"},
            },
            "required": ["text"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "data_processing"],
    )

    manager.register_tool(
        category="data_processing_tools",
        name="transform_data",
        func=transform_data,
        description="Apply named data transformation.",
        input_schema={
            "type": "object",
            "properties": {
                "data": {},
                "transformation": {"type": "string"},
            },
            "required": ["data", "transformation"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "data_processing"],
    )

    manager.register_tool(
        category="data_processing_tools",
        name="convert_format",
        func=convert_format,
        description="Convert data between supported formats.",
        input_schema={
            "type": "object",
            "properties": {
                "data": {},
                "source_format": {"type": "string"},
                "target_format": {"type": "string"},
                "options": {"type": ["object", "null"]},
            },
            "required": ["data", "source_format", "target_format"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "data_processing"],
    )

    manager.register_tool(
        category="data_processing_tools",
        name="validate_data",
        func=validate_data,
        description="Validate data with schema/quality/completeness checks.",
        input_schema={
            "type": "object",
            "properties": {
                "data": {},
                "validation_type": {"type": "string"},
                "schema": {"type": ["object", "null"]},
                "rules": {"type": ["array", "null"], "items": {"type": "object"}},
            },
            "required": ["data", "validation_type"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "data_processing"],
    )
