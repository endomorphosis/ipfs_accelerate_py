"""Compatibility response-DAG contract for older ``ipfs_datasets_py`` trees.

The canonical datasets gitlink may predate ``ipfs_datasets_py.voice.response_dag``.
Keep local queueing available without changing that gitlink; the public datasets
implementation remains authoritative whenever it can be imported.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from string import Formatter
from types import MappingProxyType
from typing import Any

ABBY_VOICE_RESPONSE_DAG_APPEND_SCHEMA_VERSION = (
    "abby_voice_response_dag_append_v1"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SECRET_KEY_MARKERS = (
    "authorization",
    "credential",
    "password",
    "secret",
    "signature",
    "token",
)


class ResponseDAGCompatibilityError(ValueError):
    """A response-DAG value violated the compatibility contract."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ResponseDAGCompatibilityError(
            "response-DAG values must be deterministic JSON"
        ) from exc


def _stable_id(prefix: str, value: Any) -> str:
    return f"{prefix}-" + sha256(_canonical_bytes(value)).hexdigest()[:24]


def _text(value: Any, *, field_name: str, required: bool = True) -> str:
    result = str(value or "").strip()
    if required and not result:
        raise ResponseDAGCompatibilityError(
            f"{field_name} must not be empty"
        )
    return result


def _digest(value: Any, *, field_name: str) -> str:
    result = _text(value, field_name=field_name).casefold()
    if not _SHA256_RE.fullmatch(result):
        raise ResponseDAGCompatibilityError(
            f"{field_name} must be a full lowercase SHA-256"
        )
    return result


def _json_safe(value: Any, *, path: str = "value") -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, bytes | bytearray | memoryview):
        raise ResponseDAGCompatibilityError(
            f"{path} must not contain raw bytes"
        )
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            if any(marker in key.casefold() for marker in _SECRET_KEY_MARKERS):
                raise ResponseDAGCompatibilityError(
                    f"{path}.{key} must not contain credentials"
                )
            result[key] = _json_safe(item, path=f"{path}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [
            _json_safe(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _json_safe(to_dict(), path=path)
    raise ResponseDAGCompatibilityError(
        f"{path} must contain deterministic JSON values"
    )


def _mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        raw = value
    else:
        to_dict = getattr(value, "to_dict", None)
        raw = to_dict() if callable(to_dict) else None
    if not isinstance(raw, Mapping):
        raise ResponseDAGCompatibilityError(
            f"{field_name} must be a mapping"
        )
    result = _json_safe(raw, path=field_name)
    if not isinstance(result, dict):
        raise ResponseDAGCompatibilityError(
            f"{field_name} must be a mapping"
        )
    return result


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [_thaw_json(item) for item in value]
    return value


def _edge(source: str, target: str, kind: str) -> dict[str, str]:
    return {
        "id": _stable_id(
            "edge",
            {"kind": kind, "source": source, "target": target},
        ),
        "kind": kind,
        "source": source,
        "target": target,
    }


def _placeholder_names(template_text: str) -> tuple[str, ...]:
    names: list[str] = []
    try:
        for _literal, field_name, format_spec, conversion in Formatter().parse(
            template_text
        ):
            if not field_name:
                continue
            if "." in field_name or "[" in field_name:
                raise ResponseDAGCompatibilityError(
                    "template fields must be simple slot names"
                )
            if format_spec or conversion:
                raise ResponseDAGCompatibilityError(
                    "template slots must not use conversions or format specs"
                )
            if field_name not in names:
                names.append(field_name)
    except ValueError as exc:
        raise ResponseDAGCompatibilityError(
            f"template_text has invalid slot syntax: {exc}"
        ) from exc
    return tuple(names)


def _normalize_audio_descriptor(value: Any) -> dict[str, Any]:
    audio = _mapping(value, field_name="audio_descriptor")
    content_sha = _digest(
        audio.get("content_sha256"),
        field_name="audio_descriptor.content_sha256",
    )
    byte_length = audio.get("byte_length")
    if (
        isinstance(byte_length, bool)
        or not isinstance(byte_length, int)
        or byte_length <= 0
    ):
        raise ResponseDAGCompatibilityError(
            "audio_descriptor.byte_length must be a positive integer"
        )
    media_type = _text(
        audio.get("media_type") or audio.get("mime_type"),
        field_name="audio_descriptor.media_type",
    )
    if not media_type.startswith("audio/"):
        raise ResponseDAGCompatibilityError(
            "audio_descriptor.media_type must be audio/*"
        )
    uri = _text(
        audio.get("uri"),
        field_name="audio_descriptor.uri",
        required=False,
    )
    ipfs_cid = _text(
        audio.get("ipfs_cid"),
        field_name="audio_descriptor.ipfs_cid",
        required=False,
    )
    if not uri and not ipfs_cid:
        raise ResponseDAGCompatibilityError(
            "validated audio requires an external uri or ipfs_cid"
        )
    if any(character.isspace() for character in uri):
        raise ResponseDAGCompatibilityError(
            "audio_descriptor.uri must not contain whitespace"
        )
    audio_id = _text(
        audio.get("audio_id")
        or _stable_id("audio", {"content_sha256": content_sha}),
        field_name="audio_descriptor.audio_id",
    )
    return {
        "audio_id": audio_id,
        "byte_length": byte_length,
        "content_sha256": content_sha,
        "id": audio_id,
        "ipfs_cid": ipfs_cid,
        "kind": "audio",
        "media_type": media_type,
        "uri": uri,
    }


def _normalize_slot_bindings(
    values: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], ...]:
    result: list[dict[str, Any]] = []
    for raw_name, raw_binding in sorted((values or {}).items()):
        name = _text(raw_name, field_name="slot_name")
        if isinstance(raw_binding, Mapping) and "value" in raw_binding:
            value = _json_safe(
                raw_binding.get("value"),
                path=f"slot.{name}.value",
            )
            source_cids_raw = raw_binding.get("source_cids") or ()
        else:
            value = _json_safe(raw_binding, path=f"slot.{name}.value")
            source_cids_raw = ()
        if value in (None, "", [], {}):
            raise ResponseDAGCompatibilityError(
                f"slot binding {name!r} must not be empty"
            )
        if isinstance(source_cids_raw, str):
            source_cids_raw = (source_cids_raw,)
        if not isinstance(source_cids_raw, Sequence):
            raise ResponseDAGCompatibilityError(
                f"slot binding {name!r} source_cids must be a sequence"
            )
        source_cids = sorted(
            {
                _text(cid, field_name=f"slot.{name}.source_cid")
                for cid in source_cids_raw
            }
        )
        node_id = _stable_id(
            "vocabulary",
            {
                "slot_name": name,
                "source_cids": source_cids,
                "value": value,
            },
        )
        result.append(
            {
                "id": node_id,
                "kind": "vocabulary",
                "slot_name": name,
                "source_cids": source_cids,
                "value": value,
                "vocabulary_id": node_id,
            }
        )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class ResponseDAGAppendCandidate:
    """Immutable append candidate compatible with the datasets contract."""

    cache_miss_event_id: str
    validation_receipt_id: str
    nodes: tuple[Mapping[str, Any], ...]
    edges: tuple[Mapping[str, Any], ...]
    rendered_text_sha256: str
    output_audio_sha256: str
    schema_version: str = ABBY_VOICE_RESPONSE_DAG_APPEND_SCHEMA_VERSION
    candidate_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        event_id = _text(
            self.cache_miss_event_id,
            field_name="cache_miss_event_id",
        )
        validation_id = _text(
            self.validation_receipt_id,
            field_name="validation_receipt_id",
        )
        rendered_digest = _digest(
            self.rendered_text_sha256,
            field_name="rendered_text_sha256",
        )
        audio_digest = _digest(
            self.output_audio_sha256,
            field_name="output_audio_sha256",
        )
        if (
            self.schema_version
            != ABBY_VOICE_RESPONSE_DAG_APPEND_SCHEMA_VERSION
        ):
            raise ResponseDAGCompatibilityError(
                f"unsupported response-DAG append schema: {self.schema_version}"
            )
        nodes = tuple(
            sorted(
                (
                    _mapping(node, field_name=f"nodes[{index}]")
                    for index, node in enumerate(self.nodes)
                ),
                key=lambda node: str(node.get("id") or ""),
            )
        )
        edges = tuple(
            sorted(
                (
                    _mapping(edge, field_name=f"edges[{index}]")
                    for index, edge in enumerate(self.edges)
                ),
                key=lambda edge: str(edge.get("id") or ""),
            )
        )
        if not nodes or not edges:
            raise ResponseDAGCompatibilityError(
                "response-DAG append requires nodes and edges"
            )
        node_ids = [
            _text(node.get("id"), field_name="node.id") for node in nodes
        ]
        edge_ids = [
            _text(edge.get("id"), field_name="edge.id") for edge in edges
        ]
        if len(node_ids) != len(set(node_ids)):
            raise ResponseDAGCompatibilityError(
                "response-DAG nodes must have unique IDs"
            )
        if len(edge_ids) != len(set(edge_ids)):
            raise ResponseDAGCompatibilityError(
                "response-DAG edges must have unique IDs"
            )
        node_kinds = [str(node.get("kind") or "") for node in nodes]
        unsupported = sorted(
            set(node_kinds)
            - {"audio", "response", "template", "vocabulary"}
        )
        if unsupported:
            raise ResponseDAGCompatibilityError(
                "response-DAG contains unsupported node kinds: "
                + ", ".join(unsupported)
            )
        if node_kinds.count("response") != 1 or node_kinds.count("audio") != 1:
            raise ResponseDAGCompatibilityError(
                "response-DAG append requires exactly one response and one audio node"
            )
        if "vocabulary" in node_kinds and node_kinds.count("template") != 1:
            raise ResponseDAGCompatibilityError(
                "vocabulary rows require exactly one slotted template row"
            )
        known_nodes = set(node_ids)
        for edge in edges:
            if (
                edge.get("source") not in known_nodes
                or edge.get("target") not in known_nodes
            ):
                raise ResponseDAGCompatibilityError(
                    f"edge {edge.get('id')!r} references an unknown node"
                )
        metadata = _mapping(self.metadata, field_name="metadata")
        object.__setattr__(self, "cache_miss_event_id", event_id)
        object.__setattr__(self, "validation_receipt_id", validation_id)
        object.__setattr__(self, "rendered_text_sha256", rendered_digest)
        object.__setattr__(self, "output_audio_sha256", audio_digest)
        object.__setattr__(
            self,
            "nodes",
            tuple(_freeze_json(node) for node in nodes),
        )
        object.__setattr__(
            self,
            "edges",
            tuple(_freeze_json(edge) for edge in edges),
        )
        object.__setattr__(self, "metadata", _freeze_json(metadata))
        computed = _stable_id("response-dag-candidate", self.identity_dict())
        if self.candidate_id and self.candidate_id != computed:
            raise ResponseDAGCompatibilityError(
                "candidate_id does not match deterministic DAG content"
            )
        object.__setattr__(self, "candidate_id", computed)

    def identity_dict(self) -> dict[str, Any]:
        return {
            "cache_miss_event_id": self.cache_miss_event_id,
            "edges": [_thaw_json(edge) for edge in self.edges],
            "metadata": _thaw_json(self.metadata),
            "nodes": [_thaw_json(node) for node in self.nodes],
            "output_audio_sha256": self.output_audio_sha256,
            "rendered_text_sha256": self.rendered_text_sha256,
            "schema_version": self.schema_version,
            "validation_receipt_id": self.validation_receipt_id,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_dict()
        payload["candidate_id"] = self.candidate_id
        payload["append_only"] = True
        return payload

    @property
    def template_rows(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            _thaw_json(node)
            for node in self.nodes
            if node.get("kind") == "template"
        )

    @property
    def vocabulary_rows(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            _thaw_json(node)
            for node in self.nodes
            if node.get("kind") == "vocabulary"
        )


def append_response_dag_candidate(
    cache_miss_event: Any,
    *,
    response_text: str,
    audio_descriptor: Any,
    template_text: str = "",
    slot_bindings: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ResponseDAGAppendCandidate:
    """Build one deterministic append from a validated cache-miss event."""

    event = _mapping(cache_miss_event, field_name="cache_miss_event")
    if event.get("ready_for_dag_append") is not True:
        raise ResponseDAGCompatibilityError(
            "cache miss must pass ASR validation before DAG append"
        )
    event_id = _text(
        event.get("event_id"),
        field_name="cache_miss_event.event_id",
    )
    validation_id = _text(
        event.get("validation_receipt_id"),
        field_name="cache_miss_event.validation_receipt_id",
    )
    rendered = _text(response_text, field_name="response_text")
    rendered_digest = sha256(rendered.encode("utf-8")).hexdigest()
    if rendered_digest != _digest(
        event.get("rendered_text_sha256"),
        field_name="cache_miss_event.rendered_text_sha256",
    ):
        raise ResponseDAGCompatibilityError(
            "response_text does not match cache-miss rendered_text_sha256"
        )
    audio_node = _normalize_audio_descriptor(audio_descriptor)
    event_audio_digest = _digest(
        event.get("output_audio_sha256"),
        field_name="cache_miss_event.output_audio_sha256",
    )
    if audio_node["content_sha256"] != event_audio_digest:
        raise ResponseDAGCompatibilityError(
            "audio descriptor does not match cache-miss output_audio_sha256"
        )
    intent = _text(
        event.get("intent") or "general",
        field_name="cache_miss_event.intent",
    )
    bindings = _normalize_slot_bindings(slot_bindings)
    binding_names = tuple(node["slot_name"] for node in bindings)
    normalized_template = _text(
        template_text,
        field_name="template_text",
        required=False,
    )
    placeholders = (
        _placeholder_names(normalized_template)
        if normalized_template
        else ()
    )
    if normalized_template and set(placeholders) != set(binding_names):
        raise ResponseDAGCompatibilityError(
            "template placeholders must exactly match slot bindings"
        )
    if bindings and not normalized_template:
        raise ResponseDAGCompatibilityError(
            "slot bindings require a reusable slotted template"
        )
    if normalized_template:
        rendered_from_template = normalized_template.format_map(
            {
                str(binding["slot_name"]): binding["value"]
                for binding in bindings
            }
        ).strip()
        if rendered_from_template != rendered:
            raise ResponseDAGCompatibilityError(
                "slotted template and vocabulary do not render response_text"
            )
    supplied_template_id = _text(
        event.get("template_id"),
        field_name="cache_miss_event.template_id",
        required=False,
    )
    template_id = supplied_template_id
    if normalized_template and not template_id:
        template_id = _stable_id(
            "template",
            {"intent": intent, "template_text": normalized_template},
        )
    response_id = _text(
        event.get("response_id"),
        field_name="cache_miss_event.response_id",
        required=False,
    ) or _stable_id(
        "response",
        {"intent": intent, "response_text": rendered},
    )
    response_node = {
        "cache_miss_event_id": event_id,
        "id": response_id,
        "intent": intent,
        "kind": "response",
        "response_id": response_id,
        "slot_names": list(binding_names),
        "spoken_text": rendered,
        "template_id": template_id,
        "text": rendered,
        "text_sha256": rendered_digest,
        "validation_receipt_id": validation_id,
    }
    nodes: list[Mapping[str, Any]] = [response_node, audio_node]
    edges: list[Mapping[str, Any]] = [
        _edge(response_id, audio_node["audio_id"], "response_to_audio")
    ]
    if template_id:
        nodes.append(
            {
                "id": template_id,
                "intent": intent,
                "kind": "template",
                "slot_names": list(placeholders),
                "spoken_template": normalized_template or None,
                "template_id": template_id,
                "template_text": normalized_template or None,
            }
        )
        edges.append(
            _edge(template_id, response_id, "template_to_response")
        )
        for vocabulary_node in bindings:
            nodes.append(vocabulary_node)
            edges.extend(
                (
                    _edge(
                        template_id,
                        str(vocabulary_node["id"]),
                        "template_to_vocabulary",
                    ),
                    _edge(
                        str(vocabulary_node["id"]),
                        response_id,
                        "vocabulary_to_response",
                    ),
                )
            )
    return ResponseDAGAppendCandidate(
        cache_miss_event_id=event_id,
        validation_receipt_id=validation_id,
        nodes=tuple(nodes),
        edges=tuple(edges),
        rendered_text_sha256=rendered_digest,
        output_audio_sha256=event_audio_digest,
        metadata=dict(metadata or {}),
    )


__all__ = [
    "ABBY_VOICE_RESPONSE_DAG_APPEND_SCHEMA_VERSION",
    "ResponseDAGAppendCandidate",
    "ResponseDAGCompatibilityError",
    "append_response_dag_candidate",
]
