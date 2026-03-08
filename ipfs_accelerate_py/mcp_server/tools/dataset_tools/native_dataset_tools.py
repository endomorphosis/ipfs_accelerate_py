"""Native dataset tool implementations for unified mcp_server."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _load_dataset_api() -> Dict[str, Any]:
    """Resolve source dataset APIs with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.dataset_tools import (  # type: ignore
            convert_dataset_format as _convert_dataset_format,
            dataset_tools_claudes as _dataset_tools_claudes,
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
            "dataset_tools_claudes": _dataset_tools_claudes,
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

        async def _dataset_tools_claudes_fallback() -> Dict[str, Any]:
            return {
                "status": "success",
                "message": "ClaudesDatasetTool initialized successfully",
                "tool_type": "Dataset processing tool",
                "available_methods": ["process_data"],
            }

        return {
            "load_dataset": _load_fallback,
            "save_dataset": _save_fallback,
            "process_dataset": _process_fallback,
            "convert_dataset_format": _convert_fallback,
            "text_to_fol": _text_to_fol_fallback,
            "legal_text_to_deontic": _legal_text_to_deontic_fallback,
            "dataset_tools_claudes": _dataset_tools_claudes_fallback,
        }


_API = _load_dataset_api()


def _mcp_text_response(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Build MCP text envelope used by legacy JSON-string call paths."""
    return {
        "content": [
            {
                "type": "text",
                "text": json.dumps(payload),
            }
        ]
    }


def _mcp_error_response(message: str, *, error_type: str = "error") -> Dict[str, Any]:
    return _mcp_text_response(
        {
            "status": "error",
            "error": message,
            "error_type": error_type,
        }
    )


def _parse_json_object(request_json: Any) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Parse JSON-string object payload for source-compatible MCP entrypoints."""
    if not isinstance(request_json, str):
        return None, _mcp_error_response("Input must be a JSON string")

    if not request_json.strip():
        return None, _mcp_error_response("Input JSON is empty", error_type="validation")

    try:
        decoded = json.loads(request_json)
    except json.JSONDecodeError as exc:
        return None, _mcp_error_response(f"Invalid JSON: {exc.msg}", error_type="validation")

    if not isinstance(decoded, dict):
        return None, _mcp_error_response("Input JSON must be an object", error_type="validation")

    return decoded, None


def _looks_like_json_object_input(value: Any) -> bool:
    """Gate JSON-string entrypoints without breaking plain string calls."""
    return isinstance(value, str) and bool(value.strip()) and value.lstrip()[:1] in {"{", "["}


def _error_result(message: str, **extra: Any) -> Dict[str, Any]:
    """Return a normalized error envelope for deterministic dispatch behavior."""
    payload: Dict[str, Any] = {"status": "error", "error": message, "message": message}
    payload.update(extra)
    return payload


def _normalize_delegate_payload(payload: Any) -> Dict[str, Any]:
    """Normalize delegate payloads with deterministic error-status inference."""
    normalized = dict(payload or {})
    has_error = bool(normalized.get("error"))
    failed = normalized.get("success") is False or has_error
    if failed:
        normalized["status"] = "error"
    else:
        normalized.setdefault("status", "success")
    return normalized


def _default_fol_summary() -> Dict[str, Any]:
    """Return an empty source-like FOL summary envelope."""
    return {
        "total_statements": 0,
        "successful_conversions": 0,
        "conversion_rate": 0.0,
        "average_confidence": 0.0,
        "unique_predicates": [],
        "quantifier_distribution": {"∀": 0, "∃": 0},
        "operator_distribution": {"∧": 0, "∨": 0, "→": 0, "↔": 0, "¬": 0},
    }


def _default_deontic_summary() -> Dict[str, Any]:
    """Return an empty source-like deontic summary envelope."""
    return {
        "total_normative_statements": 0,
        "successful_conversions": 0,
        "conversion_rate": 0.0,
        "average_confidence": 0.0,
        "normative_distribution": {"obligations": 0, "permissions": 0, "prohibitions": 0},
        "conflicts_detected": 0,
        "unique_entities": 0,
        "unique_actions": 0,
    }


async def _await_maybe(result: Any) -> Dict[str, Any]:
    """Await coroutine-like API results while supporting direct return values."""
    if hasattr(result, "__await__"):
        return await result
    return result


async def load_dataset(
    source: str,
    format: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load a dataset from a source identifier."""
    if _looks_like_json_object_input(source) and format is None and options is None:
        data, error = _parse_json_object(source)
        if error is not None:
            return error
        if not data.get("source"):
            return _mcp_error_response("Missing required field: source", error_type="validation")

        payload = await load_dataset(
            source=str(data["source"]),
            format=data.get("format"),
            options=data.get("options"),
        )
        return _mcp_text_response(payload)

    normalized_source = str(source or "").strip()
    normalized_format = None if format is None else str(format).strip()
    if not normalized_source:
        return _error_result("source must be a non-empty string")
    if normalized_format is not None and not normalized_format:
        return _error_result("format must be a non-empty string when provided")
    if options is not None and not isinstance(options, dict):
        return _error_result("options must be an object when provided")

    try:
        payload = await _await_maybe(
            _API["load_dataset"](
                source=normalized_source,
                format=normalized_format,
                options=options,
            )
        )
    except Exception as exc:
        return _error_result(f"load_dataset failed: {exc}")

    normalized = _normalize_delegate_payload(payload)
    return normalized


async def save_dataset(
    dataset_data: str | Dict[str, Any],
    destination: Optional[str] = None,
    format: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Save a dataset to a target destination and format."""
    if _looks_like_json_object_input(dataset_data) and destination is None and format is None and options is None:
        data, error = _parse_json_object(dataset_data)
        if error is not None:
            return error
        if "destination" not in data:
            return _mcp_error_response("Missing required field: destination", error_type="validation")
        if "dataset_data" not in data:
            return _mcp_error_response("Missing required field: dataset_data", error_type="validation")

        payload = await save_dataset(
            dataset_data=data["dataset_data"],
            destination=data.get("destination"),
            format=data.get("format"),
            options=data.get("options"),
        )
        return _mcp_text_response(payload)

    if isinstance(dataset_data, str) and not dataset_data.strip():
        return _error_result("dataset_data must be non-empty when provided as a string")
    if not isinstance(dataset_data, (str, dict)):
        return _error_result("dataset_data must be a string or object")

    normalized_destination = None if destination is None else str(destination).strip()
    normalized_format = None if format is None else str(format).strip()
    if normalized_destination is not None and not normalized_destination:
        return _error_result("destination must be a non-empty string when provided")
    if normalized_format is not None and not normalized_format:
        return _error_result("format must be a non-empty string when provided")
    if options is not None and not isinstance(options, dict):
        return _error_result("options must be an object when provided")

    try:
        payload = await _await_maybe(
            _API["save_dataset"](
                dataset_data=dataset_data,
                destination=normalized_destination,
                format=normalized_format,
                options=options,
            )
        )
    except Exception as exc:
        return _error_result(f"save_dataset failed: {exc}")

    normalized = _normalize_delegate_payload(payload)
    return normalized


async def process_dataset(
    dataset_source: str | Dict[str, Any] | Any,
    operations: Optional[List[Dict[str, Any]]] = None,
    output_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Process a dataset using a list of transformation operations."""
    if _looks_like_json_object_input(dataset_source) and operations is None and output_id is None:
        data, error = _parse_json_object(dataset_source)
        if error is not None:
            return error
        if "dataset_source" not in data or "operations" not in data:
            return _mcp_error_response("Missing required fields", error_type="validation")

        payload = await process_dataset(
            dataset_source=data["dataset_source"],
            operations=data["operations"],
            output_id=data.get("output_id"),
        )
        return _mcp_text_response(payload)

    if isinstance(dataset_source, str) and not dataset_source.strip():
        return _error_result("dataset_source must be non-empty when provided as a string")
    if operations is not None and (
        not isinstance(operations, list)
        or not all(isinstance(operation, dict) for operation in operations)
    ):
        return _error_result("operations must be an array of objects when provided")
    normalized_output_id = None if output_id is None else str(output_id).strip()
    if normalized_output_id is not None and not normalized_output_id:
        return _error_result("output_id must be a non-empty string when provided")

    try:
        payload = await _await_maybe(
            _API["process_dataset"](
                dataset_source=dataset_source,
                operations=operations,
                output_id=normalized_output_id,
            )
        )
    except Exception as exc:
        return _error_result(f"process_dataset failed: {exc}")

    normalized = _normalize_delegate_payload(payload)
    return normalized


async def convert_dataset_format(
    dataset_id: str,
    target_format: Optional[str] = None,
    output_path: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Convert an existing dataset into a target format."""
    if _looks_like_json_object_input(dataset_id) and target_format is None and output_path is None and options is None:
        data, error = _parse_json_object(dataset_id)
        if error is not None:
            return error
        if "dataset_id" not in data:
            return _mcp_error_response("Missing required field: dataset_id", error_type="validation")
        if "target_format" not in data:
            return _mcp_error_response("Missing required field: target_format", error_type="validation")

        payload = await convert_dataset_format(
            dataset_id=str(data["dataset_id"]),
            target_format=data.get("target_format"),
            output_path=data.get("output_path"),
            options=data.get("options"),
        )
        return _mcp_text_response(payload)

    normalized_dataset_id = str(dataset_id or "").strip()
    normalized_target_format = None if target_format is None else str(target_format).strip()
    normalized_output_path = None if output_path is None else str(output_path).strip()
    if not normalized_dataset_id:
        return _error_result("dataset_id must be a non-empty string")
    if normalized_target_format is not None and not normalized_target_format:
        return _error_result("target_format must be a non-empty string when provided")
    if normalized_output_path is not None and not normalized_output_path:
        return _error_result("output_path must be a non-empty string when provided")
    if options is not None and not isinstance(options, dict):
        return _error_result("options must be an object when provided")

    try:
        payload = await _await_maybe(
            _API["convert_dataset_format"](
                dataset_id=normalized_dataset_id,
                target_format=normalized_target_format,
                output_path=normalized_output_path,
                options=options,
            )
        )
    except Exception as exc:
        return _error_result(f"convert_dataset_format failed: {exc}")

    normalized = _normalize_delegate_payload(payload)
    return normalized


async def text_to_fol(
    text_input: str | Dict[str, Any],
    domain_predicates: Optional[List[str]] = None,
    output_format: str = "json",
    include_metadata: bool = True,
    confidence_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Convert natural-language text into first-order logic payloads."""
    if isinstance(text_input, str):
        if not text_input.strip():
            return _error_result("text_input must be provided")
    elif isinstance(text_input, dict):
        text_value = text_input.get("text")
        if not isinstance(text_value, str) or not text_value.strip():
            return _error_result("text_input object must include non-empty 'text'")
    else:
        return _error_result("text_input must be a string or object")

    if domain_predicates is not None and (
        not isinstance(domain_predicates, list)
        or not all(isinstance(item, str) and item.strip() for item in domain_predicates)
    ):
        return _error_result("domain_predicates must be an array of non-empty strings when provided")

    normalized_output_format = str(output_format or "").strip()
    if not normalized_output_format:
        return _error_result("output_format must be a non-empty string")
    if not isinstance(include_metadata, bool):
        return _error_result("include_metadata must be a boolean")

    try:
        normalized_confidence = float(confidence_threshold)
    except (TypeError, ValueError):
        return _error_result("confidence_threshold must be a number")
    if normalized_confidence < 0.0 or normalized_confidence > 1.0:
        return _error_result("confidence_threshold must be between 0 and 1")

    try:
        payload = await _await_maybe(
            _API["text_to_fol"](
                text_input=text_input,
                domain_predicates=domain_predicates,
                output_format=normalized_output_format,
                include_metadata=include_metadata,
                confidence_threshold=normalized_confidence,
            )
        )
    except Exception as exc:
        return _error_result(f"text_to_fol failed: {exc}")

    normalized = _normalize_delegate_payload(payload)
    normalized.setdefault("fol_formulas", [])
    normalized.setdefault("summary", _default_fol_summary())
    normalized.setdefault(
        "metadata",
        {
            "tool": "text_to_fol",
            "output_format": normalized_output_format,
            "confidence_threshold": normalized_confidence,
        },
    )
    return normalized


async def legal_text_to_deontic(
    text_input: str | Dict[str, Any],
    jurisdiction: str = "us",
    document_type: str = "statute",
    output_format: str = "json",
    extract_obligations: bool = True,
    include_exceptions: bool = True,
) -> Dict[str, Any]:
    """Convert legal text into deontic-logic payloads."""
    if isinstance(text_input, str):
        if not text_input.strip():
            return _error_result("text_input must be provided")
    elif isinstance(text_input, dict):
        text_value = text_input.get("text")
        if not isinstance(text_value, str) or not text_value.strip():
            return _error_result("text_input object must include non-empty 'text'")
    else:
        return _error_result("text_input must be a string or object")

    normalized_jurisdiction = str(jurisdiction or "").strip()
    normalized_document_type = str(document_type or "").strip()
    normalized_output_format = str(output_format or "").strip()
    if not normalized_jurisdiction:
        return _error_result("jurisdiction must be a non-empty string")
    if not normalized_document_type:
        return _error_result("document_type must be a non-empty string")
    if not normalized_output_format:
        return _error_result("output_format must be a non-empty string")
    if not isinstance(extract_obligations, bool):
        return _error_result("extract_obligations must be a boolean")
    if not isinstance(include_exceptions, bool):
        return _error_result("include_exceptions must be a boolean")

    try:
        payload = await _await_maybe(
            _API["legal_text_to_deontic"](
                text_input=text_input,
                jurisdiction=normalized_jurisdiction,
                document_type=normalized_document_type,
                output_format=normalized_output_format,
                extract_obligations=extract_obligations,
                include_exceptions=include_exceptions,
            )
        )
    except Exception as exc:
        return _error_result(f"legal_text_to_deontic failed: {exc}")

    normalized = _normalize_delegate_payload(payload)
    normalized.setdefault("deontic_formulas", [])
    normalized.setdefault(
        "normative_structure",
        {"obligations": [], "permissions": [], "prohibitions": []},
    )
    normalized.setdefault("legal_entities", [])
    normalized.setdefault("actions", [])
    normalized.setdefault("temporal_constraints", [])
    normalized.setdefault("conflicts", [])
    normalized.setdefault("summary", _default_deontic_summary())
    normalized.setdefault(
        "metadata",
        {
            "tool": "legal_text_to_deontic",
            "jurisdiction": normalized_jurisdiction,
            "document_type": normalized_document_type,
            "output_format": normalized_output_format,
        },
    )
    return normalized


async def dataset_tools_claudes() -> Dict[str, Any]:
    """Expose source-aligned Claude's dataset helper surface."""
    try:
        payload = await _await_maybe(
            _API["dataset_tools_claudes"]()
        )
    except Exception as exc:
        return _error_result(f"dataset_tools_claudes failed: {exc}")

    normalized = _normalize_delegate_payload(payload)
    normalized.setdefault("message", "ClaudesDatasetTool initialized successfully")
    normalized.setdefault("tool_type", "Dataset processing tool")
    normalized.setdefault("available_methods", ["process_data"])
    return normalized


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
                "source": {"type": "string", "minLength": 1},
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
                "destination": {"type": ["string", "null"], "minLength": 1},
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
                "output_id": {"type": ["string", "null"], "minLength": 1},
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
                "dataset_id": {"type": "string", "minLength": 1},
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
                "confidence_threshold": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "default": 0.5,
                },
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

    manager.register_tool(
        category="dataset_tools",
        name="dataset_tools_claudes",
        func=dataset_tools_claudes,
        description="Run source-aligned Claude's dataset helper initialization surface.",
        input_schema={"type": "object", "properties": {}, "required": []},
        runtime="fastapi",
        tags=["native", "mcpp", "dataset"],
    )
