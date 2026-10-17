"""Select literal sources from one complete selected world semantic artifact.

The caller owns final-prompt and packet provenance. This pure parser checks
content integrity of the selected reference records and their complete chunk
population, then reads only declared literal ``raw_sources`` in that artifact.
It does not authenticate producers, replay semantic admission, recover omitted
sources, or treat arbitrary reference summaries as source files. Returned text
is private intermediate data for the caller's content-free observation builder.
"""
from __future__ import annotations

import hashlib
import json
import math
import re

from ..context.context_contracts import ContextReference, canonical_context_json_bytes
from ..prompt.delta_task_packet import PROVIDER_FACING_PACKET_SCHEMA
from . import rendered_input_census as census
from .repository_world_packet import SCHEMA as WORLD_SCHEMA


WORLD_SCOPE = "world_packet.selected_semantic_context.raw_sources"
MAX_REFERENCES = 64
MAX_SOURCES = 64
MAX_MANIFEST_ITEMS = 4096
MAX_INPUT_BYTES = 256_000
WORKER_SCHEMAS = frozenset({"supervisor-semantic-worker-context@1", "supervisor-semantic-worker-context@2"})
_MEMBER_FIELDS = frozenset({"item_id", "kind", "digest", "summary", "path", "payload", "expansion_handle"})
_CHUNK_FIELDS = frozenset({"required", "priority", "chunk_index", "chunk_count", "artifact_content_id", "coverage_ids"})
_WORKER_FIELDS = frozenset({"schema", "task_id", "objective", "scope_cid", "semantic_root_cid",
    "manifest", "required_raw_paths", "preparation_bounds", "doctor_snapshot_id", "doctor_manifest_cid",
    "reconstruction", "refresh_lineage", "completion_authority", "capsules", "admissions", "raw_sources", "pack"})


def _need(value):
    if not value:
        raise ValueError("invalid_world_source_carrier")


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _bounded(value):
    """Bound builtin trees before serialization, including repeated aliases."""
    pending, nodes, text_bytes = [(value, 0)], 0, 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        _need(nodes <= census.MAX_NODES and depth <= census.MAX_DEPTH)
        if type(item) is dict:
            _need(len(item) <= census.MAX_NODES - nodes and len(pending) + len(item) <= census.MAX_NODES)
            for key, child in item.items():
                _need(type(key) is str and len(key) <= MAX_INPUT_BYTES)
                text_bytes += len(key.encode("utf-8"))
                _need(text_bytes <= MAX_INPUT_BYTES)
                pending.append((child, depth + 1))
        elif type(item) is list:
            _need(len(item) <= census.MAX_NODES - nodes and len(pending) + len(item) <= census.MAX_NODES)
            pending.extend((child, depth + 1) for child in item)
        elif type(item) is str:
            _need(len(item) <= MAX_INPUT_BYTES)
            text_bytes += len(item.encode("utf-8"))
        elif type(item) is int:
            _need(item.bit_length() <= 4096)
            text_bytes += max(1, (item.bit_length() * 30103) // 100000 + 2)
        elif type(item) is float:
            _need(math.isfinite(item))
        else:
            _need(item is None or type(item) is bool)
        _need(text_bytes <= MAX_INPUT_BYTES)
    _need(len(_json(value)) <= MAX_INPUT_BYTES)


def _path(value):
    _need(type(value) is str and 0 < len(value) <= 1024
          and len(value.encode("utf-8")) <= 1024 and "\\" not in value
          and not any(ord(char) < 32 or ord(char) == 127 for char in value)
          and len(value.split("/")) <= 64
          and all(part not in {"", ".", "..", ".git"} for part in value.split("/")))
    return value


def _source_slots(payload):
    schema = payload.get("schema")
    if type(schema) is not str or schema not in WORKER_SCHEMAS:
        raise ValueError("unsupported_world_reference_source_carrier")
    fields = _WORKER_FIELDS | ({"program_paths"} if schema.endswith("@2") else set())
    optional = {"worker_projection"} | ({"task_data_projection"} if schema.endswith("@2") else set())
    _need(fields <= set(payload) and set(payload) <= fields | optional
          and payload["completion_authority"] is False)
    sources, manifest, required = payload["raw_sources"], payload["manifest"], payload["required_raw_paths"]
    _need(type(sources) is dict and type(manifest) is dict and type(required) is list
          and len(sources) <= MAX_SOURCES and len(manifest) <= MAX_MANIFEST_ITEMS
          and len(required) <= MAX_SOURCES and set(sources) <= set(manifest))
    _need(all(type(path) is str for path in required)
          and len(required) == len(set(required)) and set(required) <= set(sources))
    if schema.endswith("@2"):
        program = payload["program_paths"]
        _need(type(program) is list and len(program) <= MAX_MANIFEST_ITEMS
              and all(type(path) is str for path in program)
              and len(program) == len(set(program)) and set(program) <= set(manifest))
    for path, binding in manifest.items():
        _path(path)
        _need(type(binding) is dict and set(binding) == {"sha256", "source_cid"}
              and type(binding["sha256"]) is str and re.fullmatch(r"[a-f0-9]{64}", binding["sha256"]) is not None
              and type(binding["source_cid"]) is str and 0 < len(binding["source_cid"].encode()) <= 4096)
    for path, content in sources.items():
        _need(type(content) is str and _sha(content.encode("utf-8")) == manifest[path]["sha256"])
    return sources, manifest, WORLD_SCOPE


def world_sources(envelope):
    """Return literal slots only after checking their one complete carrier.

    Input must be the parsed actual selected world envelope, never the native
    capsule or an out-of-band artifact. Unknown schemas, ambiguous artifacts,
    incomplete chunks and any malformed source refuse the entire population.
    Empty ``raw_sources`` is zero only inside an otherwise supported artifact.
    """
    try:
        _bounded(envelope)
        _need(type(envelope) is dict and set(envelope) == {"schema", "delivery_mode", "packet"}
              and envelope["schema"] == WORLD_SCHEMA and envelope["delivery_mode"] == "self_contained")
        packet = envelope["packet"]
        _need(type(packet) is dict and packet.get("schema") == PROVIDER_FACING_PACKET_SCHEMA
              and type(packet.get("members")) is list and 0 < len(packet["members"]) <= 512
              and all(packet.get(key) is False for key in (
                  "semantic_authority", "write_authority", "completion_authority", "contains_credentials", "contains_secrets")))
        refs, member_ids, reference_ids = [], set(), set()
        for member in packet["members"]:
            _need(type(member) is dict and set(member) == _MEMBER_FIELDS
                  and type(member["item_id"]) is str and member["item_id"] not in member_ids
                  and type(member["payload"]) is dict)
            member_ids.add(member["item_id"])
            body = member["payload"]
            if "reference" not in body:
                continue
            raw = body["reference"]
            _need(type(raw) is dict)
            if raw.get("kind") != "semantic-context":
                continue
            _need(len(refs) < MAX_REFERENCES and member["kind"] == "evidence"
                  and set(body) == {"evidence_id", "summary", "reference"}
                  and member["summary"] == body["summary"] == "Selected context reference"
                  and member["path"] == member["expansion_handle"] == "")
            reference = ContextReference.from_dict(raw)
            _need(_json(raw) == _json(reference.to_record())
                  and reference.reference_id not in reference_ids
                  and body["evidence_id"] == "context-reference:" + reference.reference_id)
            if not reference.repository_id or not reference.tree_id or not reference.path:
                raise ValueError("unsupported_world_reference_source_carrier")
            _need(reference.repository_id == packet.get("repository_id")
                  and reference.tree_id == packet.get("tree_id")
                  and len(reference.path.encode("utf-8")) <= 4096)
            # ContextReference already checks its artifact locator according
            # to the producer contract. It is not a selected source-map path.
            digest_body = {key: member[key] for key in ("kind", "summary", "path", "payload")}
            _need(member["digest"] == "sha256:" + _sha(_json(digest_body)))
            metadata = reference.metadata
            _need(set(metadata) == _CHUNK_FIELDS and metadata["required"] is True
                  and reference.tier.value == "invariant" and type(metadata["priority"]) is int
                  and type(metadata["chunk_index"]) is int and type(metadata["chunk_count"]) is int
                  and 0 <= metadata["chunk_index"] < metadata["chunk_count"] <= MAX_REFERENCES
                  and reference.byte_count == len(reference.summary.encode("utf-8"))
                  and reference.referenced_content_id == "sha256:" + _sha(reference.summary.encode("utf-8")))
            reference_ids.add(reference.reference_id)
            refs.append(reference)
        if not refs:
            raise ValueError("unsupported_world_reference_source_carrier")
        refs.sort(key=lambda reference: reference.metadata["chunk_index"])
        first = refs[0]
        prefix = first.reference_id.rsplit(":", 1)[0]
        text = "".join(reference.summary for reference in refs)
        artifact = "sha256:" + _sha(canonical_context_json_bytes({"text": text}))
        for index, reference in enumerate(refs):
            _need(reference.path == first.path and reference.metadata["chunk_index"] == index
                  and reference.reference_id == f"{prefix}:{index + 1:04d}"
                  and reference.metadata["chunk_count"] == len(refs)
                  and reference.metadata["artifact_content_id"] == artifact
                  and reference.metadata["coverage_ids"] == first.metadata["coverage_ids"]
                  and reference.metadata["priority"] == first.metadata["priority"])
        payload, prefix_text, suffix = census._json_prefix(text)
        _need(not suffix and prefix_text == text)
        _bounded(payload)
        _need(_json(payload).decode("utf-8") == text)
        return _source_slots(payload)
    except (ValueError, TypeError, KeyError, AttributeError, UnicodeError, RecursionError, OverflowError) as error:
        if type(error) is ValueError and str(error) == "unsupported_world_reference_source_carrier":
            raise ValueError("unsupported_world_reference_source_carrier") from None
        raise ValueError("invalid_world_source_carrier") from None
