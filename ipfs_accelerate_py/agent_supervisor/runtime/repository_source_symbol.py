"""Join a current native symbol to a bounded Python definition source view.

The source owner supplies a current index row and exact retained file bytes.
Native names are descriptive, not edit selectors: a byte-column AST span and
kind select the definition, then the existing editor validates its local name.
No imports, decorators, or target code execute during this operation.
"""
from __future__ import annotations

import ast
import hashlib

from .code_source_view import (
    MAX_SOURCE_BYTES, MAX_SYMBOLS, SourceViewError, _decode, _parse, _path,
    _symbols, render_source_bytes,
)


_SPAN_SCHEMA = "ipfs-datasets.software-contracts.semantic-source-span@1"
_SPAN_FIELDS = {"schema", "path", "start_line", "start_column", "end_line", "end_column"}
_COORDINATES = ("start_line", "start_column", "end_line", "end_column")
_DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
_KINDS = {
    "function": (ast.FunctionDef,),
    "async_function": (ast.AsyncFunctionDef,),
    "method": (ast.FunctionDef, ast.AsyncFunctionDef),
    "property": (ast.FunctionDef, ast.AsyncFunctionDef),
    "test": (ast.FunctionDef, ast.AsyncFunctionDef),
    "fixture": (ast.FunctionDef, ast.AsyncFunctionDef),
    "class": (ast.ClassDef,),
    "dataclass": (ast.ClassDef,),
    "typed_dict": (ast.ClassDef,),
    "enum": (ast.ClassDef,),
}


def _all_definitions(tree):
    """Include conditional duplicates when checking local selector ambiguity."""
    result = {}
    count = 0
    pending = [(tree, "")]
    while pending:
        node, prefix = pending.pop()
        if isinstance(node, _DEFINITIONS):
            prefix += node.name
            result.setdefault(prefix, []).append(node)
            count += 1
            if count > MAX_SYMBOLS:
                raise SourceViewError("symbol_limit")
            prefix += "."
        pending.extend((child, prefix) for child in ast.iter_child_nodes(node))
    return result


def _coordinates(node):
    return (node.lineno, node.col_offset, node.end_lineno, node.end_col_offset)


def render_indexed_symbol(path: str, data: bytes, row: dict, representation: str) -> dict:
    """Render an exact indexed definition or refuse with a content-free code.

    ``original`` preserves the selected definition's text after dedenting; it
    is not an entire original file. Surrounding imports, enclosing classes,
    adjacent comments and other definitions are omitted. Request a whole-file
    original read for that context, or when this supported subset refuses a
    symbol. A compact selection is only delivered when its bytes are smaller;
    the MCP owner separately compares complete response token counts.
    """
    path = _path(path)
    if representation not in {"original", "compact"}:
        raise SourceViewError("invalid_representation")
    if type(data) is not bytes or len(data) > MAX_SOURCE_BYTES:
        raise SourceViewError("source_limit")
    digest = hashlib.sha256(data).hexdigest()
    if (type(row) is not dict or row.get("path") != path
            or row.get("source_sha256") != digest):
        raise SourceViewError("indexed_symbol_source_differs")
    symbol_id, kind, span = row.get("symbol_id"), row.get("kind"), row.get("span")
    if (type(symbol_id) is not str or not symbol_id or len(symbol_id.encode("utf-8")) > 256
            or type(kind) is not str):
        raise SourceViewError("indexed_symbol_source_differs")
    if kind not in _KINDS:
        raise SourceViewError("unsupported_or_missing_symbol")
    if (type(span) is not dict or set(span) != _SPAN_FIELDS or span.get("schema") != _SPAN_SCHEMA
            or span.get("path") != path
            or any(type(span.get(key)) is not int or span[key] < (1 if key.endswith("line") else 0)
                   for key in _COORDINATES)):
        raise SourceViewError("indexed_symbol_source_differs")
    coordinates = tuple(span[key] for key in _COORDINATES)
    if coordinates[2:] < coordinates[:2]:
        raise SourceViewError("indexed_symbol_source_differs")
    source, _encoding = _decode(data)
    tree = _parse(source)
    definitions = _all_definitions(tree)
    selected = [(name, node) for name, nodes in definitions.items() for node in nodes
                if isinstance(node, _KINDS[kind]) and _coordinates(node) == coordinates]
    if len(selected) != 1:
        # A native TypedDict may be an assignment; that is outside this edit
        # subset even though its span is valid and source-current.
        if any(_coordinates(node) == coordinates for node in ast.walk(tree)
               if isinstance(node, (ast.Assign, ast.AnnAssign))):
            raise SourceViewError("unsupported_or_missing_symbol")
        raise SourceViewError("indexed_symbol_source_differs")
    symbol, node = selected[0]
    allowed = _symbols(tree)
    if symbol == "module" or node not in allowed.get(symbol, []):
        raise SourceViewError("unsupported_or_missing_symbol")
    parts = symbol.split(".")
    if any(len(definitions.get(".".join(parts[:length]), [])) != 1
           for length in range(1, len(parts) + 1)):
        raise SourceViewError("ambiguous_symbol")
    original = render_source_bytes(path, data, symbol, "raw")
    chosen = original
    if representation == "compact":
        compact = render_source_bytes(path, data, symbol, "compact")
        if (compact["format"] == "compact"
                and compact["content_bytes"] < original["content_bytes"]):
            chosen = compact
    actual = "compact" if chosen["format"] == "compact" else "original"
    return dict(path=path, original_sha256=digest, symbol_id=symbol_id, symbol=symbol,
                representation=actual, encoding="utf-8", content=chosen["content"],
                edit_supported=chosen["edit_supported"], scope="python_definition",
                limitations=["surrounding_context_omitted", "definition_dedented",
                             *(["comments_and_formatting_omitted"] if actual == "compact" else [])],
                proof_authority=False)
