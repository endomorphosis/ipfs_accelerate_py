"""Read one exact instruction carrier from an already verified final input.

This pure decoder observes bytes actually present at the typed suffix boundary.
It performs no source restoration, substring search, signature verification or
I/O. The caller owns final-input provenance and all returned source validation.
Only the simple public instruction block is supported; preceding guidance and
blocks extended with IntentIR or inventory metadata remain unavailable here.
"""
from __future__ import annotations

import hashlib
import re

from . import semantic_instruction_view as instruction
from . import semantic_metadata_view as metadata


VIEW_SCHEMA = instruction.VIEW_SCHEMA
MAX_SUFFIX_BYTES = 256_000
MAX_SOURCES = 64
_REFERENCE_FIELDS = frozenset({"schema", "source_path", "source_sha256", "source_bytes", "block_sha256"})


def _refuse():
    raise ValueError("invalid_instruction_carrier")


def _digest(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def instruction_sources(wire, suffix):
    """Return detached logical source slots including the literal relocated body.

    Other literal slots are retained once, without restoring the native prompt.
    Their content hashes and population bounds remain the parent observer's
    responsibility. Trailing reply text is outside this source inventory.
    """
    if (type(wire) is not dict or set(wire) != metadata._VIEW_FIELDS | {"instruction_instructions"}
            or wire.get("schema") != VIEW_SCHEMA
            or wire.get("source_transport_schema") != metadata.TRANSPORT_SCHEMA
            or wire.get("instruction_instructions") != instruction.INSTRUCTIONS
            or wire.get("metadata_instructions") != metadata.METADATA_INSTRUCTIONS
            or wire.get("execution_authority") is not False or wire.get("completion_authority") is not False
            or any(type(wire.get(key)) is not dict for key in
                   ("translated_semantic", "translation_table", "native_context", "common_bindings"))
            or type(wire.get("instructions")) is not str or type(wire.get("native_suffix")) is not str
            or type(suffix) is not str or len(suffix) > MAX_SUFFIX_BYTES):
        _refuse()
    semantic = wire["translated_semantic"]
    sources, manifest = semantic.get("raw_sources"), semantic.get("manifest")
    if type(sources) is not dict or not 1 <= len(sources) <= MAX_SOURCES or type(manifest) is not dict:
        _refuse()
    result, selected = {}, None
    for path, content in sources.items():
        if type(path) is not str:
            _refuse()
        if type(content) is str:
            result[path] = content
            continue
        if (selected is not None or type(content) is not dict or set(content) != {"$public_instruction"}
                or type(content["$public_instruction"]) is not dict):
            _refuse()
        reference = content["$public_instruction"]
        if (set(reference) != _REFERENCE_FIELDS or reference.get("schema") != instruction.REFERENCE_SCHEMA
                or reference.get("source_path") != path or not path or path.startswith("/")
                or "\\" in path or "\0" in path or any(part in {"", ".", ".."} for part in path.split("/"))
                or not _digest(reference.get("source_sha256")) or not _digest(reference.get("block_sha256"))
                or type(reference.get("source_bytes")) is not int or not 1 <= reference["source_bytes"] <= 32_768
                or type(manifest.get(path)) is not dict
                or manifest[path].get("sha256") != reference["source_sha256"]):
            _refuse()
        selected = path, reference
    if selected is None:
        _refuse()
    path, reference = selected
    try:
        raw = suffix.encode("utf-8")
    except UnicodeError:
        _refuse()
    prefix, ending = instruction._PREFIX.encode("utf-8"), instruction._END.encode("utf-8")
    start, end = len(prefix), len(prefix) + reference["source_bytes"]
    block_end = end + len(ending)
    if (len(raw) > MAX_SUFFIX_BYTES or not raw.startswith(prefix) or raw[end:block_end] != ending
            or hashlib.sha256(raw[start:end]).hexdigest() != reference["source_sha256"]
            or hashlib.sha256(raw[:block_end]).hexdigest() != reference["block_sha256"]):
        _refuse()
    try:
        source = raw[start:end].decode("utf-8")
    except UnicodeError:
        _refuse()
    if not source.strip() or "\0" in source:
        _refuse()
    result[path] = source
    return result


__all__ = ["VIEW_SCHEMA", "instruction_sources"]
