"""MCP-IDL registry primitives for unified MCP++ runtime.

This module provides deterministic descriptor canonicalization and a small
in-memory interface repository used by Profile A (`mcp-idl`) tools.
"""

from __future__ import annotations

import base64
import binascii
import copy
from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


AI_CATALOG_INTERFACE_NAME = "ai.catalog.v1"
AI_CATALOG_NAMESPACE = "ipfs_accelerate_py.ai.catalog"
AI_CATALOG_VERSION = "1.0.0"
AI_CATALOG_SCHEMA_REVISION = "ai.catalog.idl.v1"
AI_CATALOG_READ_AUTHORITY = "ai.catalog/read"
AI_CATALOG_REFRESH_AUTHORITY = "ai.catalog/refresh"
AI_CATALOG_INVOKE_AUTHORITY = "ai.catalog/invoke"

MAX_CATALOG_PAGE_SIZE = 1_000
MAX_CATALOG_SOURCES = 64
MAX_CATALOG_CURSOR_BYTES = 4_096
MAX_SELECTOR_BYTES = 256
MAX_OPERATION_BYTES = 64
MAX_DIAGNOSTICS = 64
MAX_DIAGNOSTIC_MESSAGE_BYTES = 1_024
MAX_STREAM_CHUNKS = 1_024
MAX_TIMEOUT_SECONDS = 120.0
MAX_TEXT_ITEM_BYTES = 262_144
MAX_INPUT_ITEMS = 128
MAX_EMBEDDING_DIMENSIONS = 16_384
MAX_TEXT_EMBEDDING_OUTPUT_BYTES = 4_194_304
MAX_MEDIA_OUTPUT_BYTES = 8_388_608
MAX_INLINE_MEDIA_BYTES = 8_388_608
MAX_MEDIA_BYTES = 33_554_432
MAX_MEDIA_DURATION_SECONDS = 600.0
MAX_SAMPLE_RATE_HZ = 192_000
MIN_SAMPLE_RATE_HZ = 8_000
MAX_IMAGE_DIMENSION = 16_384
MAX_IMAGE_PIXELS = 40_000_000
MAX_URI_BYTES = 2_048
MAX_ARTIFACT_REF_BYTES = 1_024
MAX_POLICY_ENTRIES = 64
MAX_JSON_DEPTH = 16
MAX_JSON_PROPERTIES = 256
MAX_JSON_ITEMS = 1_000
MAX_JSON_STRING_BYTES = 262_144
MAX_JSON_KEY_BYTES = 256
MAX_JSON_NUMBER_ABS = 10**308
MAX_ENVELOPE_OVERHEAD_BYTES = 1_048_576


def _normalize_capability(value: Any) -> str:
    """Normalize capability tokens for tolerant compatibility matching.

    Normalization rules are intentionally conservative:
    - trim whitespace
    - lowercase
    - drop optional version suffix after `@`
    """

    text = str(value or "").strip().lower()
    if "@" in text:
        text = text.split("@", 1)[0].strip()
    return text


def _missing_capabilities(requires: Iterable[Any], supported: Iterable[Any]) -> list[str]:
    """Return required capabilities that are not satisfied by supported set."""

    supported_norm = {_normalize_capability(x) for x in supported if _normalize_capability(x)}
    missing: list[str] = []
    for req in requires:
        req_text = str(req or "").strip()
        req_norm = _normalize_capability(req_text)
        if not req_norm:
            continue
        if req_norm not in supported_norm:
            missing.append(req_text)
    return sorted(missing)


def canonicalize_descriptor(descriptor: Dict[str, Any]) -> bytes:
    """Return deterministic canonical bytes for an interface descriptor."""
    return json.dumps(descriptor, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def compute_interface_cid(descriptor: Dict[str, Any]) -> str:
    """Compute stable content-addressed identifier for descriptor bytes.

    The implementation uses a deterministic SHA-256 digest representation as a
    lightweight CID placeholder for current migration phases.
    """
    digest = hashlib.sha256(canonicalize_descriptor(descriptor)).hexdigest()
    return f"cidv1-sha256-{digest}"


def build_descriptor(
    *,
    name: str,
    namespace: str,
    version: str,
    methods: List[Dict[str, Any]],
    errors: Optional[List[Dict[str, Any]]] = None,
    requires: Optional[List[str]] = None,
    compatibility: Optional[Dict[str, List[str]]] = None,
    semantic_tags: Optional[List[str]] = None,
    observability: Optional[Dict[str, Any]] = None,
    interaction_patterns: Optional[Dict[str, Any]] = None,
    resource_cost_hints: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a descriptor with required and optional MCP-IDL fields."""
    return {
        "name": name,
        "namespace": namespace,
        "version": version,
        "methods": methods,
        "errors": errors or [],
        "requires": requires or [],
        "compatibility": compatibility or {"compatible_with": [], "supersedes": []},
        "semantic_tags": semantic_tags or [],
        "observability": observability or {"trace": True, "provenance": True},
        "interaction_patterns": interaction_patterns or {"request_response": True, "event_streams": False},
        "resource_cost_hints": resource_cost_hints or {},
    }


def _object_schema(
    properties: Mapping[str, Any],
    required: Sequence[str] = (),
    *,
    max_properties: Optional[int] = None,
) -> Dict[str, Any]:
    schema: Dict[str, Any] = {
        "type": "object",
        "properties": copy.deepcopy(dict(properties)),
        "required": list(required),
        "additionalProperties": False,
    }
    if max_properties is not None:
        schema["maxProperties"] = max_properties
    return schema


def _nullable(schema: Mapping[str, Any]) -> Dict[str, Any]:
    return {"oneOf": [copy.deepcopy(dict(schema)), {"type": "null"}]}


def _bounded_string(max_length: int, *, min_length: int = 0) -> Dict[str, Any]:
    schema: Dict[str, Any] = {
        "type": "string",
        "maxLength": max_length,
        "x-maxUtf8Bytes": max_length,
    }
    if min_length:
        schema["minLength"] = min_length
    return schema


def _bounded_json_schema() -> Dict[str, Any]:
    """Return a reference to the output envelope's bounded JSON definition."""
    return {"$ref": "#/$defs/boundedJson"}


def _bounded_json_definitions() -> Dict[str, Any]:
    return {
        "boundedJson": {
            "oneOf": [
                {"type": "null"},
                {"type": "boolean"},
                {"type": "number"},
                _bounded_string(MAX_JSON_STRING_BYTES),
                {
                    "type": "array",
                    "maxItems": MAX_JSON_ITEMS,
                    "items": {"$ref": "#/$defs/boundedJson"},
                },
                {
                    "type": "object",
                    "maxProperties": MAX_JSON_PROPERTIES,
                    "additionalProperties": {"$ref": "#/$defs/boundedJson"},
                },
            ],
            # JSON Schema has no portable recursion-depth keyword. MCP++
            # transports enforce this extension before decoding records.
            "x-maxDepth": MAX_JSON_DEPTH,
        }
    }


def _catalog_filter_input_schema() -> Dict[str, Any]:
    """Mirror the local ``model_catalog_list_*`` MCP input schema."""
    return _object_schema(
        {
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_CATALOG_PAGE_SIZE,
                "default": 100,
            },
            "cursor": {
                "type": "string",
                "minLength": 1,
                "maxLength": MAX_CATALOG_CURSOR_BYTES,
            },
            "provider": {
                "type": "string",
                "minLength": 1,
                "maxLength": MAX_SELECTOR_BYTES,
            },
            "model": {
                "type": "string",
                "minLength": 1,
                "maxLength": MAX_SELECTOR_BYTES,
            },
            "operation": {
                "type": "string",
                "minLength": 1,
                "maxLength": MAX_OPERATION_BYTES,
            },
            "modality": {
                "type": "string",
                "minLength": 1,
                "maxLength": MAX_OPERATION_BYTES,
            },
            "state": {
                "type": "object",
                "maxProperties": 6,
                "additionalProperties": {"type": "boolean"},
            },
            "labels": {
                "type": "object",
                "maxProperties": 64,
                "additionalProperties": {
                    "type": "string",
                    "maxLength": MAX_SELECTOR_BYTES,
                },
            },
        }
    )


def _policy_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "maxProperties": MAX_POLICY_ENTRIES,
        "additionalProperties": {"type": ["string", "number", "boolean"]},
    }


def _invocation_constraints(*, output_bytes: int) -> Dict[str, Any]:
    return {
        "service": {
            "type": "string",
            "minLength": 1,
            "maxLength": MAX_SELECTOR_BYTES,
        },
        "model": {
            "type": "string",
            "minLength": 1,
            "maxLength": MAX_SELECTOR_BYTES,
        },
        "provider": {
            "type": "string",
            "minLength": 1,
            "maxLength": MAX_SELECTOR_BYTES,
        },
        "policy": _policy_schema(),
        "device": {
            "type": "string",
            "minLength": 1,
            "maxLength": MAX_SELECTOR_BYTES,
        },
        "timeout": {
            "type": "number",
            "exclusiveMinimum": 0,
            "maximum": MAX_TIMEOUT_SECONDS,
            "default": 30.0,
        },
        "max_output_bytes": {
            "type": "integer",
            "minimum": 1,
            "maximum": output_bytes,
            "default": output_bytes,
        },
        "allow_fallback": {"type": "boolean", "default": False},
        "stream": {"type": "boolean", "default": False},
        "max_stream_chunks": {
            "type": "integer",
            "minimum": 1,
            "maximum": MAX_STREAM_CHUNKS,
            "default": MAX_STREAM_CHUNKS,
        },
    }


_IMAGE_MIME_TYPES = ("image/gif", "image/jpeg", "image/png", "image/webp")
_AUDIO_MIME_TYPES = (
    "audio/aac",
    "audio/flac",
    "audio/mp3",
    "audio/mp4",
    "audio/mpeg",
    "audio/ogg",
    "audio/wav",
    "audio/webm",
    "audio/x-wav",
)


def _media_schema(media_kind: str) -> Dict[str, Any]:
    """Mirror the local vision/voice discriminated media input schema."""
    mime_types = list(_IMAGE_MIME_TYPES if media_kind == "image" else _AUDIO_MIME_TYPES)
    metadata = (
        {
            "width": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_IMAGE_DIMENSION,
            },
            "height": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_IMAGE_DIMENSION,
            },
        }
        if media_kind == "image"
        else {
            "duration_seconds": {
                "type": "number",
                "exclusiveMinimum": 0,
                "maximum": MAX_MEDIA_DURATION_SECONDS,
            },
            "sample_rate_hz": {
                "type": "integer",
                "minimum": MIN_SAMPLE_RATE_HZ,
                "maximum": MAX_SAMPLE_RATE_HZ,
            },
        }
    )
    required_metadata = list(metadata)

    def variant(
        source: str,
        specific: Mapping[str, Any],
        required: Sequence[str],
    ) -> Dict[str, Any]:
        return _object_schema(
            {
                "source": {"const": source},
                "mime_type": {"type": "string", "enum": mime_types},
                "byte_length": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_MEDIA_BYTES,
                },
                **metadata,
                **specific,
            },
            ["source", "mime_type", *required_metadata, *required],
        )

    return {
        "oneOf": [
            variant(
                "inline",
                {
                    "data_base64": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": ((MAX_INLINE_MEDIA_BYTES + 2) // 3) * 4,
                    }
                },
                ["data_base64"],
            ),
            variant(
                "uri",
                {
                    "uri": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": MAX_URI_BYTES,
                        "pattern": "^https://",
                    }
                },
                ["uri", "byte_length"],
            ),
            variant(
                "artifact",
                {
                    "artifact_ref": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": MAX_ARTIFACT_REF_BYTES,
                    }
                },
                ["artifact_ref", "byte_length"],
            ),
        ]
    }


def ai_catalog_v1_input_schemas() -> Dict[str, Dict[str, Any]]:
    """Return MCP-parity input schemas keyed by canonical local tool name."""
    resolve = _object_schema(
        {
            "operation": {
                "type": "string",
                "minLength": 1,
                "maxLength": MAX_OPERATION_BYTES,
            },
            "modality": {"type": "string", "maxLength": MAX_OPERATION_BYTES},
            "model": {"type": "string", "maxLength": MAX_SELECTOR_BYTES},
            "provider": {"type": "string", "maxLength": MAX_SELECTOR_BYTES},
            "deployment": {"type": "string", "maxLength": MAX_SELECTOR_BYTES},
            "policy": _policy_schema(),
            "device": {"type": "string", "maxLength": MAX_SELECTOR_BYTES},
            "context": {
                "type": "integer",
                "minimum": 1,
                "maximum": 100_000_000,
            },
            "health": {"type": "boolean"},
            "locality": {"type": "string", "maxLength": MAX_SELECTOR_BYTES},
            "configured": {"type": "boolean"},
            "authorized": {"type": "boolean"},
            "reachable": {"type": "boolean"},
            "routable": {"type": "boolean"},
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_CATALOG_PAGE_SIZE,
                "default": 100,
            },
        },
        ["operation"],
    )
    text = _invocation_constraints(output_bytes=MAX_TEXT_EMBEDDING_OUTPUT_BYTES)
    text.update(
        {
            "prompt": {
                "type": "string",
                "minLength": 1,
                "maxLength": MAX_TEXT_ITEM_BYTES,
            },
            "max_tokens": {
                "type": "integer",
                "minimum": 1,
                "maximum": 1_000_000,
                "default": 512,
            },
            "temperature": {
                "type": "number",
                "minimum": 0,
                "maximum": 2,
                "default": 0.7,
            },
        }
    )
    embeddings = _invocation_constraints(
        output_bytes=MAX_TEXT_EMBEDDING_OUTPUT_BYTES
    )
    embeddings.update(
        {
            "texts": {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_INPUT_ITEMS,
                "items": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_TEXT_ITEM_BYTES,
                },
            },
            "dimensions": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_EMBEDDING_DIMENSIONS,
            },
        }
    )
    multimodal = _invocation_constraints(output_bytes=MAX_MEDIA_OUTPUT_BYTES)
    multimodal.update(
        {
            "prompt": {
                "type": "string",
                "minLength": 1,
                "maxLength": MAX_TEXT_ITEM_BYTES,
            },
            "media": {
                "type": "array",
                "minItems": 1,
                "maxItems": 1,
                "items": _media_schema("image"),
            },
            "max_tokens": {
                "type": "integer",
                "minimum": 1,
                "maximum": 1_000_000,
                "default": 512,
            },
            "temperature": {
                "type": "number",
                "minimum": 0,
                "maximum": 2,
                "default": 0.7,
            },
            "allow_remote_media": {"type": "boolean", "default": False},
        }
    )
    transcribe = _invocation_constraints(output_bytes=MAX_MEDIA_OUTPUT_BYTES)
    transcribe.update(
        {
            "audio": _media_schema("audio"),
            "language": {
                "type": "string",
                "minLength": 1,
                "maxLength": MAX_SELECTOR_BYTES,
            },
            "allow_remote_media": {"type": "boolean", "default": False},
        }
    )
    synthesize = _invocation_constraints(output_bytes=MAX_MEDIA_OUTPUT_BYTES)
    synthesize.update(
        {
            "text": {
                "type": "string",
                "minLength": 1,
                "maxLength": MAX_TEXT_ITEM_BYTES,
            },
            "output_mime_type": {
                "type": "string",
                "enum": list(_AUDIO_MIME_TYPES),
                "default": "audio/wav",
            },
            "voice": {
                "type": "string",
                "minLength": 1,
                "maxLength": MAX_SELECTOR_BYTES,
            },
            "sample_rate_hz": {
                "type": "integer",
                "minimum": MIN_SAMPLE_RATE_HZ,
                "maximum": MAX_SAMPLE_RATE_HZ,
                "default": 24_000,
            },
            "max_duration_seconds": {
                "type": "number",
                "exclusiveMinimum": 0,
                "maximum": MAX_MEDIA_DURATION_SECONDS,
                "default": 120.0,
            },
        }
    )
    return {
        "model_catalog_list_services": _catalog_filter_input_schema(),
        "model_catalog_list_models": _catalog_filter_input_schema(),
        "model_catalog_get": _object_schema(
            {
                "identifier": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_SELECTOR_BYTES,
                },
                "record_type": {
                    "type": "string",
                    "enum": [
                        "records",
                        "providers",
                        "models",
                        "deployments",
                        "bindings",
                    ],
                },
            },
            ["identifier"],
        ),
        "model_catalog_resolve": resolve,
        "model_catalog_health": _object_schema({}),
        "model_catalog_refresh": _object_schema(
            {
                "sources": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": MAX_CATALOG_SOURCES,
                    "uniqueItems": True,
                    "items": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 128,
                    },
                },
                "authority": {
                    "type": "boolean",
                    "const": True,
                    "description": "Explicit authorization for this named refresh.",
                },
            },
            ["sources", "authority"],
        ),
        "llm_generate": _object_schema(text, ["prompt"]),
        "embeddings_generate": _object_schema(embeddings, ["texts"]),
        "multimodal_generate": _object_schema(multimodal, ["prompt", "media"]),
        "voice_transcribe": _object_schema(transcribe, ["audio"]),
        "voice_synthesize": _object_schema(synthesize, ["text"]),
    }


def _diagnostic_schema() -> Dict[str, Any]:
    return _object_schema(
        {
            "code": _bounded_string(MAX_OPERATION_BYTES, min_length=1),
            "message": _bounded_string(
                MAX_DIAGNOSTIC_MESSAGE_BYTES, min_length=1
            ),
            "severity": {
                "type": "string",
                "enum": ["info", "warning", "error"],
            },
            "source": _nullable(_bounded_string(128, min_length=1)),
            "record_type": _nullable(_bounded_string(32, min_length=1)),
            "record_id": _nullable(
                _bounded_string(MAX_SELECTOR_BYTES, min_length=1)
            ),
            "field": _nullable(
                _bounded_string(MAX_SELECTOR_BYTES, min_length=1)
            ),
            "winner_source": _nullable(_bounded_string(128, min_length=1)),
            "ambiguous": {"type": "boolean"},
            "context": _bounded_json_schema(),
        },
        ["code", "message"],
        max_properties=10,
    )


def _diagnostics_schema() -> Dict[str, Any]:
    return {
        "type": "array",
        "maxItems": MAX_DIAGNOSTICS,
        "items": _diagnostic_schema(),
    }


def _error_schema() -> Dict[str, Any]:
    return _object_schema(
        {
            "code": _bounded_string(MAX_OPERATION_BYTES, min_length=1),
            "message": _bounded_string(
                MAX_DIAGNOSTIC_MESSAGE_BYTES, min_length=1
            ),
            "cause": _bounded_string(128, min_length=1),
            "diagnostics": _diagnostics_schema(),
        },
        ["code", "message"],
        max_properties=4,
    )


def _streaming_result_schema() -> Dict[str, Any]:
    return _object_schema(
        {
            "requested": {"type": "boolean"},
            "supported": {"const": False},
            "mode": {"const": "buffered"},
            "max_chunks": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_STREAM_CHUNKS,
            },
        },
        ["requested", "supported", "mode", "max_chunks"],
    )


def _output_envelope(
    operation_properties: Mapping[str, Any],
    *,
    success_required: Sequence[str],
    max_serialized_bytes: int,
) -> Dict[str, Any]:
    properties: Dict[str, Any] = {
        "status": {"type": "string", "enum": ["success", "error"]},
        "success": {"type": "boolean"},
        "tool_schema_version": _bounded_string(64, min_length=1),
        "schema_version": _nullable(_bounded_string(128, min_length=1)),
        "catalog_revision": _nullable(_bounded_string(256, min_length=1)),
        "error": _error_schema(),
        "error_code": _bounded_string(MAX_OPERATION_BYTES, min_length=1),
        "error_type": _bounded_string(MAX_OPERATION_BYTES, min_length=1),
        **copy.deepcopy(dict(operation_properties)),
    }
    return {
        **_object_schema(
            properties,
            [
                "status",
                "success",
                "tool_schema_version",
                "schema_version",
                "catalog_revision",
            ],
        ),
        "$defs": _bounded_json_definitions(),
        "allOf": [
            {
                "oneOf": [
                    {
                        "properties": {
                            "status": {"const": "success"},
                            "success": {"const": True},
                            "schema_version": _bounded_string(
                                128, min_length=1
                            ),
                            "catalog_revision": _bounded_string(
                                256, min_length=1
                            ),
                        },
                        "required": [
                            "status",
                            "success",
                            *list(success_required),
                        ],
                    },
                    {
                        "properties": {
                            "status": {"const": "error"},
                            "success": {"const": False},
                        },
                        "required": ["status", "success", "error"],
                    },
                ]
            }
        ],
        # The operation payload is bounded separately. Reserve deterministic
        # space for versions, selection receipts, and diagnostics.
        "x-maxSerializedBytes": (
            max_serialized_bytes + MAX_ENVELOPE_OVERHEAD_BYTES
        ),
        "x-maxDepth": MAX_JSON_DEPTH,
    }


def ai_catalog_v1_output_schemas() -> Dict[str, Dict[str, Any]]:
    """Return bounded output records matching local MCP response envelopes."""
    bounded = _bounded_json_schema()
    page_common = {
        "items": {
            "type": "array",
            "maxItems": MAX_CATALOG_PAGE_SIZE,
            "items": bounded,
        },
        "record_type": _bounded_string(32, min_length=1),
        "count": {
            "type": "integer",
            "minimum": 0,
            "maximum": MAX_CATALOG_PAGE_SIZE,
        },
        "total": {
            "type": "integer",
            "minimum": 0,
            "maximum": 1_000_000_000,
        },
        "next_cursor": _nullable(
            _bounded_string(MAX_CATALOG_CURSOR_BYTES, min_length=1)
        ),
        "diagnostics": _diagnostics_schema(),
    }
    selected_and_receipt = {
        "selected_binding": bounded,
        "receipt": bounded,
        "streaming": _streaming_result_schema(),
        "diagnostics": _diagnostics_schema(),
    }
    return {
        "model_catalog_list_services": _output_envelope(
            {
                **page_common,
                "services": {
                    "type": "array",
                    "maxItems": MAX_CATALOG_PAGE_SIZE,
                    "items": bounded,
                },
            },
            success_required=[
                "items",
                "services",
                "record_type",
                "count",
                "total",
                "next_cursor",
            ],
            max_serialized_bytes=MAX_MEDIA_OUTPUT_BYTES,
        ),
        "model_catalog_list_models": _output_envelope(
            {
                **page_common,
                "models": {
                    "type": "array",
                    "maxItems": MAX_CATALOG_PAGE_SIZE,
                    "items": bounded,
                },
            },
            success_required=[
                "items",
                "models",
                "record_type",
                "count",
                "total",
                "next_cursor",
            ],
            max_serialized_bytes=MAX_MEDIA_OUTPUT_BYTES,
        ),
        "model_catalog_get": _output_envelope(
            {
                "record_type": _nullable(
                    _bounded_string(32, min_length=1)
                ),
                "query": _bounded_string(MAX_SELECTOR_BYTES, min_length=1),
                # boundedJson already includes JSON null.
                "record": bounded,
                "diagnostics": _diagnostics_schema(),
            },
            success_required=["record_type", "query", "record", "diagnostics"],
            max_serialized_bytes=MAX_MEDIA_OUTPUT_BYTES,
        ),
        "model_catalog_resolve": _output_envelope(
            {"resolution": bounded, "diagnostics": _diagnostics_schema()},
            success_required=["resolution"],
            max_serialized_bytes=MAX_MEDIA_OUTPUT_BYTES,
        ),
        "model_catalog_health": _output_envelope(
            {"health": bounded, "diagnostics": _diagnostics_schema()},
            success_required=["health"],
            max_serialized_bytes=MAX_MEDIA_OUTPUT_BYTES,
        ),
        "model_catalog_refresh": _output_envelope(
            {
                "refreshed": {
                    "type": "array",
                    "maxItems": MAX_CATALOG_SOURCES,
                    "items": _bounded_string(128, min_length=1),
                },
                "failed": {
                    "type": "array",
                    "maxItems": MAX_CATALOG_SOURCES,
                    "items": _bounded_string(128, min_length=1),
                },
                "unchanged": {
                    "type": "array",
                    "maxItems": MAX_CATALOG_SOURCES,
                    "items": _bounded_string(128, min_length=1),
                },
                "source_states": {
                    "type": "array",
                    "maxItems": MAX_CATALOG_SOURCES,
                    "items": bounded,
                },
                "diagnostics": _diagnostics_schema(),
            },
            success_required=[
                "refreshed",
                "failed",
                "unchanged",
                "source_states",
                "diagnostics",
            ],
            max_serialized_bytes=MAX_MEDIA_OUTPUT_BYTES,
        ),
        "llm_generate": _output_envelope(
            {
                "text": _bounded_string(MAX_TEXT_EMBEDDING_OUTPUT_BYTES),
                **selected_and_receipt,
            },
            success_required=[
                "text",
                "selected_binding",
                "receipt",
                "streaming",
            ],
            max_serialized_bytes=MAX_TEXT_EMBEDDING_OUTPUT_BYTES,
        ),
        "embeddings_generate": _output_envelope(
            {
                "embeddings": {
                    "type": "array",
                    "maxItems": MAX_INPUT_ITEMS,
                    "items": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": MAX_EMBEDDING_DIMENSIONS,
                        "items": {"type": "number"},
                    },
                },
                "count": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": MAX_INPUT_ITEMS,
                },
                "dimensions": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_EMBEDDING_DIMENSIONS,
                },
                **selected_and_receipt,
            },
            success_required=[
                "embeddings",
                "count",
                "dimensions",
                "selected_binding",
                "receipt",
                "streaming",
            ],
            max_serialized_bytes=MAX_TEXT_EMBEDDING_OUTPUT_BYTES,
        ),
        "multimodal_generate": _output_envelope(
            {
                "text": _bounded_string(MAX_MEDIA_OUTPUT_BYTES),
                **selected_and_receipt,
            },
            success_required=[
                "text",
                "selected_binding",
                "receipt",
                "streaming",
            ],
            max_serialized_bytes=MAX_MEDIA_OUTPUT_BYTES,
        ),
        "voice_transcribe": _output_envelope(
            {
                "text": _bounded_string(MAX_MEDIA_OUTPUT_BYTES),
                **selected_and_receipt,
            },
            success_required=[
                "text",
                "selected_binding",
                "receipt",
                "streaming",
            ],
            max_serialized_bytes=MAX_MEDIA_OUTPUT_BYTES,
        ),
        "voice_synthesize": _output_envelope(
            {
                "audio": _object_schema(
                    {
                        "mime_type": {
                            "type": "string",
                            "enum": list(_AUDIO_MIME_TYPES),
                        },
                        "byte_length": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": MAX_MEDIA_OUTPUT_BYTES,
                        },
                        "data_base64": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": (
                                (MAX_MEDIA_OUTPUT_BYTES + 2) // 3
                            )
                            * 4,
                            "contentEncoding": "base64",
                            "x-maxDecodedBytes": MAX_MEDIA_OUTPUT_BYTES,
                        },
                        "sha256": {
                            "type": "string",
                            "minLength": 64,
                            "maxLength": 64,
                            "pattern": "^[0-9a-f]{64}$",
                        },
                    },
                    ["mime_type", "byte_length", "data_base64", "sha256"],
                ),
                **selected_and_receipt,
            },
            success_required=[
                "audio",
                "selected_binding",
                "receipt",
                "streaming",
            ],
            # Synthesized bytes are base64 encoded in the local MCP response.
            max_serialized_bytes=(
                ((MAX_MEDIA_OUTPUT_BYTES + 2) // 3) * 4
            ),
        ),
    }


_AI_CATALOG_AUTHORITIES = {
    "model_catalog_list_services": AI_CATALOG_READ_AUTHORITY,
    "model_catalog_list_models": AI_CATALOG_READ_AUTHORITY,
    "model_catalog_get": AI_CATALOG_READ_AUTHORITY,
    "model_catalog_resolve": AI_CATALOG_READ_AUTHORITY,
    "model_catalog_health": AI_CATALOG_READ_AUTHORITY,
    "model_catalog_refresh": AI_CATALOG_REFRESH_AUTHORITY,
    "llm_generate": AI_CATALOG_INVOKE_AUTHORITY,
    "embeddings_generate": AI_CATALOG_INVOKE_AUTHORITY,
    "multimodal_generate": AI_CATALOG_INVOKE_AUTHORITY,
    "voice_transcribe": AI_CATALOG_INVOKE_AUTHORITY,
    "voice_synthesize": AI_CATALOG_INVOKE_AUTHORITY,
}


def build_ai_catalog_v1_descriptor() -> Dict[str, Any]:
    """Build the production ``ai.catalog.v1`` MCP++ interface descriptor."""
    inputs = ai_catalog_v1_input_schemas()
    outputs = ai_catalog_v1_output_schemas()
    paginated = {
        "model_catalog_list_services",
        "model_catalog_list_models",
    }
    methods: List[Dict[str, Any]] = []
    for operation, authority in _AI_CATALOG_AUTHORITIES.items():
        is_paginated = operation in paginated
        is_invocation = authority == AI_CATALOG_INVOKE_AUTHORITY
        methods.append(
            {
                "name": f"{AI_CATALOG_INTERFACE_NAME}/{operation}",
                "operation": operation,
                "mcp_tool": operation,
                "input_schema": inputs[operation],
                "output_schema": outputs[operation],
                "required_authority": authority,
                "pagination": {
                    "mode": (
                        "revision-bound-cursor"
                        if is_paginated
                        else "none"
                    ),
                    "request_cursor_field": (
                        "cursor" if is_paginated else None
                    ),
                    "response_cursor_field": (
                        "next_cursor" if is_paginated else None
                    ),
                    "limit_field": "limit" if is_paginated else None,
                    "max_page_items": (
                        MAX_CATALOG_PAGE_SIZE if is_paginated else 1
                    ),
                    "catalog_revision_field": "catalog_revision",
                    "cursor_invalidated_on_revision_change": is_paginated,
                },
                "streaming": {
                    "supported": False,
                    "mode": "buffered",
                    "request_field": "stream" if is_invocation else None,
                    "max_chunks_field": (
                        "max_stream_chunks" if is_invocation else None
                    ),
                    "max_chunks": (
                        MAX_STREAM_CHUNKS if is_invocation else 0
                    ),
                },
            }
        )
    descriptor = build_descriptor(
        name=AI_CATALOG_INTERFACE_NAME,
        namespace=AI_CATALOG_NAMESPACE,
        version=AI_CATALOG_VERSION,
        methods=methods,
        errors=[
            {
                "name": "UnknownInterfaceVersion",
                "code": "unknown_interface_version",
                "upgrade_metadata": True,
            },
            {
                "name": "UnknownOperation",
                "code": "unknown_operation",
                "upgrade_metadata": True,
            },
            {"name": "InvalidRequest", "code": "invalid_request"},
            {"name": "Unauthorized", "code": "unauthorized"},
            {"name": "ResponseTooLarge", "code": "response_too_large"},
        ],
        requires=["mcp++/profile-a-idl"],
        semantic_tags=[
            "ai-catalog",
            "model-manager",
            "bounded",
            "revisioned",
            "federation-ready",
        ],
        observability={
            "trace": True,
            "provenance": True,
            "diagnostics": {
                "max_items": MAX_DIAGNOSTICS,
                "max_message_bytes": MAX_DIAGNOSTIC_MESSAGE_BYTES,
                "max_depth": MAX_JSON_DEPTH,
            },
        },
        interaction_patterns={
            "request_response": True,
            "event_streams": False,
            "streaming": "buffered-only",
            "pagination": "revision-bound-cursor",
        },
        resource_cost_hints={
            "max_timeout_seconds": MAX_TIMEOUT_SECONDS,
            "max_page_items": MAX_CATALOG_PAGE_SIZE,
            "max_stream_chunks": MAX_STREAM_CHUNKS,
            "max_inline_media_bytes": MAX_INLINE_MEDIA_BYTES,
            "max_media_bytes": MAX_MEDIA_BYTES,
            "max_output_bytes": MAX_MEDIA_OUTPUT_BYTES,
        },
    )
    descriptor.update(
        {
            "protocol_version": "v1",
            "schema_revision": AI_CATALOG_SCHEMA_REVISION,
            "catalog_revision": {
                "location": "response.catalog_revision",
                "required": True,
                "maxLength": 256,
            },
            "authorities": {
                "read": AI_CATALOG_READ_AUTHORITY,
                "refresh": AI_CATALOG_REFRESH_AUTHORITY,
                "invoke": AI_CATALOG_INVOKE_AUTHORITY,
            },
            "transport_bounds": {
                "max_json_depth": MAX_JSON_DEPTH,
                "max_json_properties": MAX_JSON_PROPERTIES,
                "max_json_items": MAX_JSON_ITEMS,
                "max_json_string_bytes": MAX_JSON_STRING_BYTES,
                "max_json_key_bytes": MAX_JSON_KEY_BYTES,
                "max_json_number_abs": str(MAX_JSON_NUMBER_ABS),
                "max_diagnostics": MAX_DIAGNOSTICS,
                "max_timeout_seconds": MAX_TIMEOUT_SECONDS,
            },
        }
    )
    return descriptor


@dataclass
class InterfaceUpgradeRequired(ValueError):
    """Fail-closed interface or operation negotiation error."""

    code: str
    message: str
    requested_version: str
    requested_operation: str
    supported_versions: List[str]
    supported_operations: List[str]
    interface_cid: str

    def __str__(self) -> str:
        return self.message

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": "error",
            "success": False,
            "error": {
                "code": _safe_upgrade_text(
                    self.code, MAX_OPERATION_BYTES
                ),
                "message": _safe_upgrade_text(
                    self.message, MAX_DIAGNOSTIC_MESSAGE_BYTES
                ),
            },
            "upgrade": {
                "interface": AI_CATALOG_INTERFACE_NAME,
                "requested_version": _safe_upgrade_text(
                    self.requested_version, MAX_SELECTOR_BYTES
                ),
                "requested_operation": _safe_upgrade_text(
                    self.requested_operation, MAX_SELECTOR_BYTES
                ),
                "supported_versions": [
                    _safe_upgrade_text(item, MAX_SELECTOR_BYTES)
                    for item in self.supported_versions[:16]
                ],
                "supported_operations": [
                    _safe_upgrade_text(item, MAX_OPERATION_BYTES)
                    for item in self.supported_operations[:64]
                ],
                "latest_version": AI_CATALOG_VERSION,
                "interface_cid": _safe_upgrade_text(
                    self.interface_cid, MAX_SELECTOR_BYTES
                ),
                "schema_revision": AI_CATALOG_SCHEMA_REVISION,
            },
        }


def _safe_upgrade_text(value: Any, maximum_bytes: int) -> str:
    """Return bounded negotiation metadata without reflecting arbitrary objects."""
    if isinstance(value, str):
        text = value
    else:
        text = f"<{type(value).__name__}>"
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= maximum_bytes:
        return encoded.decode("utf-8")
    return encoded[:maximum_bytes].decode("utf-8", errors="ignore")


def resolve_ai_catalog_operation(
    operation: str,
    *,
    version: str = AI_CATALOG_VERSION,
) -> Dict[str, Any]:
    """Resolve one supported operation or fail closed with upgrade metadata."""
    descriptor = build_ai_catalog_v1_descriptor()
    interface_cid = compute_interface_cid(descriptor)
    methods = {
        str(method["operation"]): method
        for method in descriptor["methods"]
    }
    supported_operations = list(methods)
    if not isinstance(version, str) or version not in {
        AI_CATALOG_VERSION,
        "v1",
    }:
        raise InterfaceUpgradeRequired(
            code="unknown_interface_version",
            message=(
                "The requested ai.catalog interface version is unsupported."
            ),
            requested_version=_safe_upgrade_text(
                version, MAX_SELECTOR_BYTES
            ),
            requested_operation=_safe_upgrade_text(
                operation, MAX_SELECTOR_BYTES
            ),
            supported_versions=[AI_CATALOG_VERSION],
            supported_operations=supported_operations,
            interface_cid=interface_cid,
        )
    if not isinstance(operation, str) or operation not in methods:
        raise InterfaceUpgradeRequired(
            code="unknown_operation",
            message="The requested ai.catalog operation is unsupported.",
            requested_version=_safe_upgrade_text(
                version, MAX_SELECTOR_BYTES
            ),
            requested_operation=_safe_upgrade_text(
                operation, MAX_SELECTOR_BYTES
            ),
            supported_versions=[AI_CATALOG_VERSION],
            supported_operations=supported_operations,
            interface_cid=interface_cid,
        )
    return copy.deepcopy(methods[operation])


@dataclass
class IDLValidationError(ValueError):
    """Bounded, secret-safe payload validation failure."""

    code: str
    path: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}: {self.message}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": "error",
            "success": False,
            "error": {
                "code": self.code,
                "message": self.message,
            },
            "diagnostics": [
                {
                    "code": self.code,
                    "message": self.message,
                    "severity": "error",
                    "context": {
                        "path": self.path[:MAX_SELECTOR_BYTES]
                    },
                }
            ],
        }


def _schema_error(path: str, message: str) -> None:
    raise IDLValidationError(
        code="invalid_request",
        path=path[:MAX_SELECTOR_BYTES],
        message=message[:MAX_DIAGNOSTIC_MESSAGE_BYTES],
    )


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if isinstance(value, float) and not math.isfinite(value):
        return False
    return abs(value) <= MAX_JSON_NUMBER_ABS


def _decode_bounded_base64(
    value: str,
    path: str,
    maximum: int,
) -> bytes:
    try:
        decoded = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError, TypeError) as exc:
        _schema_error(path, "string is not valid base64")
        raise AssertionError("unreachable") from exc
    if not decoded:
        _schema_error(path, "decoded media must not be empty")
    if len(decoded) > maximum:
        _schema_error(
            path, "decoded media exceeds the maximum byte length"
        )
    return decoded


def _validate_schema_value(
    value: Any,
    schema: Mapping[str, Any],
    *,
    path: str,
    depth: int,
    definitions: Optional[Mapping[str, Any]] = None,
) -> None:
    """Validate the bounded JSON Schema subset used by this IDL."""
    if depth > MAX_JSON_DEPTH:
        _schema_error(
            path, "value exceeds the maximum JSON nesting depth"
        )
    local_definitions = schema.get("$defs", definitions)
    reference = schema.get("$ref")
    if reference is not None:
        if (
            not isinstance(reference, str)
            or not reference.startswith("#/$defs/")
            or not isinstance(local_definitions, Mapping)
        ):
            _schema_error(path, "schema contains an unsupported reference")
        name = reference.rsplit("/", 1)[-1]
        target = local_definitions.get(name)
        if not isinstance(target, Mapping):
            _schema_error(path, "schema reference could not be resolved")
        _validate_schema_value(
            value,
            target,
            path=path,
            depth=depth,
            definitions=local_definitions,
        )
        return

    conjuncts = schema.get("allOf")
    if isinstance(conjuncts, Sequence) and not isinstance(
        conjuncts, (str, bytes)
    ):
        for conjunct in conjuncts:
            if isinstance(conjunct, Mapping):
                _validate_schema_value(
                    value,
                    conjunct,
                    path=path,
                    depth=depth,
                    definitions=local_definitions,
                )

    choices = schema.get("oneOf")
    if isinstance(choices, Sequence) and not isinstance(
        choices, (str, bytes)
    ):
        matches = 0
        for choice in choices:
            if not isinstance(choice, Mapping):
                continue
            try:
                _validate_schema_value(
                    value,
                    choice,
                    path=path,
                    depth=depth,
                    definitions=local_definitions,
                )
            except IDLValidationError:
                continue
            matches += 1
        if matches != 1:
            _schema_error(
                path, "value does not match exactly one schema variant"
            )
        return

    if "const" in schema and value != schema["const"]:
        _schema_error(
            path, "value does not match the required constant"
        )
    if "enum" in schema and value not in schema["enum"]:
        _schema_error(
            path, "value is not one of the supported values"
        )

    declared = schema.get("type")
    types = list(declared) if isinstance(declared, list) else [declared]
    if declared is not None:
        valid_type = any(
            (
                item == "null"
                and value is None
                or item == "boolean"
                and isinstance(value, bool)
                or item == "integer"
                and isinstance(value, int)
                and not isinstance(value, bool)
                or item == "number"
                and _is_finite_number(value)
                or item == "string"
                and isinstance(value, str)
                or item == "array"
                and isinstance(value, Sequence)
                and not isinstance(
                    value, (str, bytes, bytearray, Mapping)
                )
                or item == "object"
                and isinstance(value, Mapping)
            )
            for item in types
        )
        if not valid_type:
            _schema_error(path, "value has an invalid type")

    if isinstance(value, str):
        if len(value) < int(schema.get("minLength", 0)):
            _schema_error(
                path, "string is shorter than the minimum length"
            )
        maximum_length = schema.get("maxLength")
        if maximum_length is None and "string" in types:
            # Local MCP policy values intentionally use a scalar union. The
            # transport supplies the shared ceiling without changing parity.
            maximum_length = MAX_JSON_STRING_BYTES
        if (
            maximum_length is not None
            and len(value) > int(maximum_length)
        ):
            _schema_error(path, "string exceeds the maximum length")
        maximum_bytes = schema.get(
            "x-maxUtf8Bytes", maximum_length
        )
        if (
            maximum_bytes is not None
            and len(value.encode("utf-8")) > int(maximum_bytes)
        ):
            _schema_error(
                path,
                "string exceeds the maximum UTF-8 byte length",
            )
        pattern = schema.get("pattern")
        if (
            isinstance(pattern, str)
            and re.search(pattern, value) is None
        ):
            _schema_error(
                path, "string does not match the required pattern"
            )
        if schema.get("contentEncoding") == "base64" or path.endswith(
            ".data_base64"
        ):
            _decode_bounded_base64(
                value,
                path,
                int(
                    schema.get(
                        "x-maxDecodedBytes", MAX_INLINE_MEDIA_BYTES
                    )
                ),
            )
        return

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not _is_finite_number(value):
            _schema_error(
                path, "number exceeds the finite transport bound"
            )
        if "minimum" in schema and value < schema["minimum"]:
            _schema_error(path, "number is below the minimum")
        if (
            "exclusiveMinimum" in schema
            and value <= schema["exclusiveMinimum"]
        ):
            _schema_error(
                path, "number is below the exclusive minimum"
            )
        if "maximum" in schema and value > schema["maximum"]:
            _schema_error(path, "number exceeds the maximum")
        if (
            "exclusiveMaximum" in schema
            and value >= schema["exclusiveMaximum"]
        ):
            _schema_error(
                path, "number exceeds the exclusive maximum"
            )
        return

    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray, Mapping))
    ):
        if len(value) < int(schema.get("minItems", 0)):
            _schema_error(
                path, "array has fewer than the minimum items"
            )
        if (
            "maxItems" in schema
            and len(value) > int(schema["maxItems"])
        ):
            _schema_error(
                path, "array exceeds the maximum item count"
            )
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value):
                _validate_schema_value(
                    item,
                    item_schema,
                    path=f"{path}[{index}]",
                    depth=depth + 1,
                    definitions=local_definitions,
                )
        if schema.get("uniqueItems"):
            try:
                canonical_items = [
                    json.dumps(
                        item,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    for item in value
                ]
            except (TypeError, ValueError, OverflowError):
                _schema_error(
                    path, "array contains a non-JSON item"
                )
            if len(canonical_items) != len(set(canonical_items)):
                _schema_error(path, "array items must be unique")
        return

    if isinstance(value, Mapping):
        if (
            "maxProperties" in schema
            and len(value) > int(schema["maxProperties"])
        ):
            _schema_error(
                path, "object exceeds the maximum property count"
            )
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if isinstance(required, Sequence):
            for name in required:
                if name not in value:
                    _schema_error(
                        path,
                        f"missing required property {name!r}",
                    )
        for name, item in value.items():
            if not isinstance(name, str):
                _schema_error(
                    path, "object property names must be strings"
                )
            if (
                len(name.encode("utf-8", errors="replace"))
                > MAX_JSON_KEY_BYTES
            ):
                _schema_error(
                    path,
                    "object property name exceeds the byte limit",
                )
            child_path = (
                f"{path}.{str(name)[:MAX_SELECTOR_BYTES]}"
            )
            child_schema = (
                properties.get(name)
                if isinstance(properties, Mapping)
                else None
            )
            if isinstance(child_schema, Mapping):
                _validate_schema_value(
                    item,
                    child_schema,
                    path=child_path,
                    depth=depth + 1,
                    definitions=local_definitions,
                )
                continue
            additional = schema.get("additionalProperties", True)
            if additional is False:
                _schema_error(
                    child_path,
                    "additional property is not allowed",
                )
            if isinstance(additional, Mapping):
                _validate_schema_value(
                    item,
                    additional,
                    path=child_path,
                    depth=depth + 1,
                    definitions=local_definitions,
                )
        encoded_media: Optional[bytes] = None
        data_base64 = value.get("data_base64")
        if isinstance(data_base64, str):
            encoded_media = _decode_bounded_base64(
                data_base64,
                f"{path}.data_base64",
                MAX_MEDIA_OUTPUT_BYTES,
            )
            declared_bytes = value.get("byte_length")
            if (
                declared_bytes is not None
                and declared_bytes != len(encoded_media)
            ):
                _schema_error(
                    f"{path}.byte_length",
                    "byte_length does not match decoded media",
                )
        digest = value.get("sha256")
        if encoded_media is not None and isinstance(digest, str):
            if hashlib.sha256(encoded_media).hexdigest() != digest:
                _schema_error(
                    f"{path}.sha256",
                    "sha256 does not match media",
                )
        width = value.get("width")
        height = value.get("height")
        if (
            isinstance(width, int)
            and not isinstance(width, bool)
            and isinstance(height, int)
            and not isinstance(height, bool)
            and width * height > MAX_IMAGE_PIXELS
        ):
            _schema_error(
                path, "image dimensions exceed the pixel limit"
            )


def validate_ai_catalog_payload(
    operation: str,
    payload: Mapping[str, Any],
    *,
    direction: str = "input",
    version: str = AI_CATALOG_VERSION,
) -> Dict[str, Any]:
    """Validate and isolate an AI catalog request or response record."""
    method = resolve_ai_catalog_operation(operation, version=version)
    if direction not in {"input", "output"}:
        raise ValueError("direction must be 'input' or 'output'")
    if not isinstance(payload, Mapping):
        _schema_error("$", "payload must be an object")
    schema = method[f"{direction}_schema"]
    if direction == "output":
        max_bytes = schema.get("x-maxSerializedBytes")
        try:
            encoded = canonicalize_descriptor(dict(payload))
        except (TypeError, ValueError, OverflowError):
            _schema_error("$", "payload is not bounded JSON")
        if isinstance(max_bytes, int) and len(encoded) > max_bytes:
            raise IDLValidationError(
                code="response_too_large",
                path="$",
                message=(
                    "response exceeds the maximum serialized byte length"
                ),
            )
    _validate_schema_value(payload, schema, path="$", depth=0)
    return copy.deepcopy(dict(payload))


def authorize_ai_catalog_operation(
    operation: str,
    granted_authorities: Iterable[str],
    *,
    version: str = AI_CATALOG_VERSION,
) -> Dict[str, Any]:
    """Resolve an operation only when its exact authority is granted."""
    method = resolve_ai_catalog_operation(operation, version=version)
    granted = {str(item).strip() for item in granted_authorities}
    required = str(method["required_authority"])
    if required not in granted:
        raise PermissionError(
            f"missing required authority: {required}"
        )
    return method


@dataclass(frozen=True)
class CompatibilityVerdict:
    """Compatibility result shape for `interfaces/compat`."""

    compatible: bool
    reasons: List[str]
    requires_missing: List[str]
    suggested_alternatives: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "compatible": self.compatible,
            "reasons": list(self.reasons),
            "requires_missing": list(self.requires_missing),
            "suggested_alternatives": list(self.suggested_alternatives),
        }


class InterfaceDescriptorRegistry:
    """In-memory MCP-IDL descriptor repository."""

    def __init__(self, supported_capabilities: Optional[Iterable[str]] = None) -> None:
        self._by_cid: Dict[str, Dict[str, Any]] = {}
        self._supported_capabilities = {str(x) for x in (supported_capabilities or [])}

    @property
    def supported_capabilities(self) -> set[str]:
        """Return supported capabilities used for compatibility checks."""
        return set(self._supported_capabilities)

    def set_supported_capabilities(self, capabilities: Iterable[str]) -> None:
        """Replace capability set used by compatibility checks."""
        self._supported_capabilities = {str(x) for x in capabilities}

    def register_descriptor(self, descriptor: Dict[str, Any]) -> str:
        """Register a descriptor and return its computed `interface_cid`."""
        interface_cid = compute_interface_cid(descriptor)
        payload = copy.deepcopy(descriptor)
        payload["interface_cid"] = interface_cid
        self._by_cid[interface_cid] = payload
        return interface_cid

    def register_ai_catalog_v1(self) -> str:
        """Register the canonical bounded AI catalog interface."""
        return register_ai_catalog_v1(self)

    def resolve_ai_catalog_operation(
        self,
        operation: str,
        *,
        version: str = AI_CATALOG_VERSION,
    ) -> Dict[str, Any]:
        """Resolve an operation only if this registry contains the interface."""
        method = resolve_ai_catalog_operation(
            operation, version=version
        )
        expected_cid = compute_interface_cid(
            build_ai_catalog_v1_descriptor()
        )
        if expected_cid not in self._by_cid:
            descriptor = build_ai_catalog_v1_descriptor()
            supported = [
                str(item["operation"])
                for item in descriptor["methods"]
            ]
            raise InterfaceUpgradeRequired(
                code="interface_not_registered",
                message=(
                    "The requested ai.catalog interface is not registered."
                ),
                requested_version=_safe_upgrade_text(
                    version, MAX_SELECTOR_BYTES
                ),
                requested_operation=_safe_upgrade_text(
                    operation, MAX_SELECTOR_BYTES
                ),
                supported_versions=[AI_CATALOG_VERSION],
                supported_operations=supported,
                interface_cid=expected_cid,
            )
        return method

    def list_interfaces(self) -> List[str]:
        """List registered interface CIDs in deterministic order."""
        return sorted(self._by_cid.keys())

    def get_descriptor(self, interface_cid: str) -> Optional[Dict[str, Any]]:
        """Get descriptor payload for CID, if present."""
        payload = self._by_cid.get(interface_cid)
        return copy.deepcopy(payload) if payload is not None else None

    def compat(self, interface_cid: str) -> CompatibilityVerdict:
        """Evaluate compatibility against local supported capabilities."""
        descriptor = self._by_cid.get(interface_cid)
        if descriptor is None:
            return CompatibilityVerdict(
                compatible=False,
                reasons=["interface_not_found"],
                requires_missing=[],
                suggested_alternatives=[],
            )

        requires = [str(x) for x in descriptor.get("requires", [])]
        missing = _missing_capabilities(requires, self._supported_capabilities)
        if missing:
            return CompatibilityVerdict(
                compatible=False,
                reasons=["missing_required_capabilities"],
                requires_missing=missing,
                suggested_alternatives=sorted(
                    [
                        cid
                        for cid, payload in self._by_cid.items()
                        if cid != interface_cid
                        and not _missing_capabilities(payload.get("requires", []), self._supported_capabilities)
                    ]
                ),
            )

        return CompatibilityVerdict(
            compatible=True,
            reasons=[],
            requires_missing=[],
            suggested_alternatives=[],
        )

    def select(self, task_hint_cid: str = "", budget: int = 20) -> List[str]:
        """Return a deterministic, budgeted subset of compatible interfaces."""
        _ = task_hint_cid  # Reserved for future ranking heuristics.
        compatible = [cid for cid in self.list_interfaces() if self.compat(cid).compatible]
        return compatible[: max(0, int(budget))]


def register_ai_catalog_v1(registry: InterfaceDescriptorRegistry) -> str:
    """Register ``ai.catalog.v1`` and return its deterministic interface CID."""
    if not isinstance(registry, InterfaceDescriptorRegistry):
        raise TypeError(
            "registry must be an InterfaceDescriptorRegistry"
        )
    return registry.register_descriptor(
        build_ai_catalog_v1_descriptor()
    )


__all__ = [
    "AI_CATALOG_INTERFACE_NAME",
    "AI_CATALOG_INVOKE_AUTHORITY",
    "AI_CATALOG_READ_AUTHORITY",
    "AI_CATALOG_REFRESH_AUTHORITY",
    "AI_CATALOG_SCHEMA_REVISION",
    "AI_CATALOG_VERSION",
    "CompatibilityVerdict",
    "IDLValidationError",
    "InterfaceUpgradeRequired",
    "InterfaceDescriptorRegistry",
    "ai_catalog_v1_input_schemas",
    "ai_catalog_v1_output_schemas",
    "authorize_ai_catalog_operation",
    "build_ai_catalog_v1_descriptor",
    "build_descriptor",
    "canonicalize_descriptor",
    "compute_interface_cid",
    "register_ai_catalog_v1",
    "resolve_ai_catalog_operation",
    "validate_ai_catalog_payload",
]
