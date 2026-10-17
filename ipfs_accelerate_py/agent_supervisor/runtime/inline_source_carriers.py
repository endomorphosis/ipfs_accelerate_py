"""Select bounded literal source slots from explicitly supported wrappers.

The caller bounds and parses the final prompt, validates every selected literal
against its manifest, and owns input provenance. These structural checks do not
replay native producers, restore omitted text, or authenticate dispatch. In
particular, source-read display rows are checked but never counted as originals.
"""
from __future__ import annotations

import json

from . import semantic_query_reference as query


QUERY_SCHEMA = query.VIEW_SCHEMA
SOURCE_INPUT_SCHEMA = "supervisor-source-input-view@1"
INSTRUCTION_SCHEMA = "supervisor-semantic-instruction-view@1"
CARRIER_SCOPES = {
    QUERY_SCHEMA: "transport.translated_semantic.raw_sources",
    SOURCE_INPUT_SCHEMA: "semantic_transport.translated_semantic.raw_sources",
    INSTRUCTION_SCHEMA: "translated_semantic.raw_sources_and_bound_public_instruction",
}
_SOURCE_FIELDS = frozenset({"schema", "semantic_transport", "source_read_view", "factored_ast",
                            "execution_authority", "completion_authority"})
_BLOCK_FIELDS = frozenset({"schema", "views", "instructions", "edit_authority", "proof_authority",
                           "completion_authority"})
_ROW_FIELDS = frozenset({"id", "path", "symbol", "source_sha256", "format", "content"})


def _need(condition):
    if not condition:
        raise ValueError("invalid_source_carrier")


def _query_sources(wire):
    _need(set(wire) == query._VIEW_FIELDS and wire.get("instructions") == query.INSTRUCTIONS
          and wire.get("execution_authority") is False and wire.get("completion_authority") is False)
    inner = wire["transport"]
    semantic, _bindings = query._transport(inner, referenced=True)
    _need(query._markers(inner) == [query.SLOT])
    path = semantic["worker_projection"]["query"][query.REF_KEY]
    sources = semantic["raw_sources"]
    _need(path in semantic["required_raw_paths"] and path in sources)
    literal = sources[path]
    _need(1 <= len(literal.encode("utf-8")) <= query.MAX_QUERY_BYTES
          and sum(sources[name] == literal for name in semantic["required_raw_paths"]) == 1)
    return sources, semantic["manifest"]


def _source_input_sources(wire):
    _need(set(wire) == _SOURCE_FIELDS and wire.get("factored_ast") is True
          and wire.get("execution_authority") is False and wire.get("completion_authority") is False)
    inner = wire["semantic_transport"]
    _need(type(inner) is dict)
    schema = inner.get("schema")
    _need(type(schema) is str and schema in {query.TRANSPORT_SCHEMA, query.COMPACT_TRANSPORT_SCHEMA})
    expected = query._BASE_FIELDS if schema == query.TRANSPORT_SCHEMA else query._COMPACT_FIELDS
    _need(set(inner) == expected and inner.get("execution_authority") is False
          and inner.get("completion_authority") is False and type(inner.get("instructions")) is str
          and type(inner.get("native_suffix")) is str)
    if schema == query.TRANSPORT_SCHEMA:
        _need(type(inner.get("translation_table")) is dict)
    semantic = inner.get("translated_semantic")
    _need(type(semantic) is dict)
    sources, manifest = semantic.get("raw_sources"), semantic.get("manifest")
    _need(type(sources) is dict and type(manifest) is dict and len(sources) <= 64
          and len(manifest) <= 4096 and set(sources) <= set(manifest))
    block = wire["source_read_view"]
    _need(type(block) is dict and set(block) == _BLOCK_FIELDS
          and block.get("schema") == "supervisor-source-read-input@1"
          and type(block.get("instructions")) is str
          and block.get("edit_authority") is False and block.get("proof_authority") is False
          and block.get("completion_authority") is False
          and type(block.get("views")) is list and 1 <= len(block["views"]) <= 8)
    _need(len(json.dumps(block, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")) <= 32768)
    for index, row in enumerate(block["views"]):
        _need(type(row) is dict and set(row) == _ROW_FIELDS and row.get("id") == f"v{index}"
              and type(row.get("format")) is str and row["format"] in {"raw", "compact"}
              and type(row.get("content")) is str)
        path = query._path(row.get("path"))
        query._text(row.get("symbol"), maximum=1024)
        digest = query._digest(row.get("source_sha256"))
        _need(path in manifest and path not in sources and type(manifest[path]) is dict
              and manifest[path].get("sha256") == digest)
    return sources, manifest


def source_carrier(wire, suffix):
    """Return (literal sources, manifest, exact slot scope), without restoration.

    ``wire`` and ``suffix`` must come from the caller's bounded JSON-prefix
    parser. Only the instruction carrier consumes its exact appended block.
    Other suffix text is outside coverage. Invalid or nested wrappers refuse as
    one population; errors contain no source values or private exception text.
    """
    try:
        _need(type(wire) is dict and type(suffix) is str)
        schema = wire.get("schema")
        _need(type(schema) is str and schema in CARRIER_SCOPES)
        if schema == QUERY_SCHEMA:
            sources, manifest = _query_sources(wire)
        elif schema == SOURCE_INPUT_SCHEMA:
            sources, manifest = _source_input_sources(wire)
        else:
            # The instruction helper binds its one relocated literal to exact
            # suffix bytes; it never reconstructs arbitrary referenced sources.
            from .inline_source_instruction import instruction_sources

            sources = instruction_sources(wire, suffix)
            manifest = wire["translated_semantic"]["manifest"]
        return sources, manifest, CARRIER_SCOPES[schema]
    except (ValueError, TypeError, KeyError, UnicodeError, RecursionError, OverflowError):
        raise ValueError("invalid_source_carrier") from None
