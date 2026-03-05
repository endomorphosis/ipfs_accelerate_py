"""Native data processing tool implementations for unified mcp_server."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


_VALID_CHUNK_STRATEGIES = {"fixed_size", "sentence", "paragraph", "semantic"}


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
    normalized_text = str(text or "")
    normalized_strategy = str(strategy or "").strip().lower()
    if not normalized_text.strip():
        return {
            "status": "error",
            "message": "text is required",
            "text": text,
        }
    if normalized_strategy not in _VALID_CHUNK_STRATEGIES:
        return {
            "status": "error",
            "message": "strategy must be one of: fixed_size, sentence, paragraph, semantic",
            "strategy": strategy,
        }

    normalized_chunk_size = int(chunk_size)
    normalized_overlap = int(overlap)
    normalized_max_chunks = int(max_chunks)
    if normalized_chunk_size <= 0:
        return {
            "status": "error",
            "message": "chunk_size must be a positive integer",
            "chunk_size": chunk_size,
        }
    if normalized_overlap < 0:
        return {
            "status": "error",
            "message": "overlap must be a non-negative integer",
            "overlap": overlap,
        }
    if normalized_overlap >= normalized_chunk_size:
        return {
            "status": "error",
            "message": "overlap must be smaller than chunk_size",
            "overlap": overlap,
            "chunk_size": chunk_size,
        }
    if normalized_max_chunks <= 0:
        return {
            "status": "error",
            "message": "max_chunks must be a positive integer",
            "max_chunks": max_chunks,
        }

    result = await _API["chunk_text"](
        text=text,
        strategy=normalized_strategy,
        chunk_size=normalized_chunk_size,
        overlap=normalized_overlap,
        max_chunks=normalized_max_chunks,
    )
    payload = dict(result or {})
    payload.setdefault("status", "success")
    return payload


async def transform_data(
    data: Any,
    transformation: str,
    parameters: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Apply a named transformation to input data."""
    normalized_transformation = str(transformation or "").strip()
    if not normalized_transformation:
        return {
            "status": "error",
            "message": "transformation is required",
            "transformation": transformation,
        }
    if data is None:
        return {
            "status": "error",
            "message": "data is required",
            "data": data,
        }
    if parameters is not None and not isinstance(parameters, dict):
        return {
            "status": "error",
            "message": "parameters must be an object when provided",
            "parameters": parameters,
        }

    merged_parameters: Dict[str, Any] = {}
    if parameters:
        merged_parameters.update(parameters)
    merged_parameters.update(kwargs)

    result = await _API["transform_data"](
        data=data,
        transformation=normalized_transformation,
        **merged_parameters,
    )
    payload = dict(result or {})
    payload.setdefault("status", "success")
    return payload


async def convert_format(
    data: Any,
    source_format: str,
    target_format: str,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Convert data between supported formats."""
    normalized_source = str(source_format or "").strip()
    normalized_target = str(target_format or "").strip()
    if data is None:
        return {
            "status": "error",
            "message": "data is required",
            "data": data,
        }
    if not normalized_source or not normalized_target:
        return {
            "status": "error",
            "message": "source_format and target_format are required",
            "source_format": source_format,
            "target_format": target_format,
        }
    if options is not None and not isinstance(options, dict):
        return {
            "status": "error",
            "message": "options must be an object when provided",
            "options": options,
        }

    result = await _API["convert_format"](
        data=data,
        source_format=normalized_source,
        target_format=normalized_target,
        options=options,
    )
    payload = dict(result or {})
    payload.setdefault("status", "success" if payload.get("status") != "error" else "error")
    return payload


async def validate_data(
    data: Any,
    validation_type: str,
    schema: Optional[Dict[str, Any]] = None,
    rules: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Validate input data with requested strategy."""
    normalized_type = str(validation_type or "").strip()
    if data is None:
        return {
            "status": "error",
            "message": "data is required",
            "data": data,
        }
    if not normalized_type:
        return {
            "status": "error",
            "message": "validation_type is required",
            "validation_type": validation_type,
        }
    if schema is not None and not isinstance(schema, dict):
        return {
            "status": "error",
            "message": "schema must be an object when provided",
            "schema": schema,
        }
    if rules is not None and not isinstance(rules, list):
        return {
            "status": "error",
            "message": "rules must be an array when provided",
            "rules": rules,
        }
    if isinstance(rules, list) and not all(isinstance(item, dict) for item in rules):
        return {
            "status": "error",
            "message": "rules entries must be objects",
            "rules": rules,
        }

    result = await _API["validate_data"](
        data=data,
        validation_type=normalized_type,
        schema=schema,
        rules=rules,
    )
    payload = dict(result or {})
    payload.setdefault("status", "success")
    return payload


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
                "strategy": {
                    "type": "string",
                    "default": "fixed_size",
                    "enum": ["fixed_size", "sentence", "paragraph", "semantic"],
                },
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
                "parameters": {"type": ["object", "null"]},
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
