"""Dependency-light helpers for grounded GraphRAG voice templates.

The voice router deliberately treats GraphRAG as a response-plan provider.  A
retriever may use an LLM, an IPLD graph, or a local test double, but it must
receive a stable query envelope and it must return a template whose factual
slots can be checked against current evidence.  This module contains the
small, dependency-free pieces shared by those boundaries; it does not import
``ipfs_datasets_py`` or initialize a model, network client, or vector store.

``buildVoiceGraphRagPromptParts`` keeps the historical camel-case name used by
the objective scanner.  ``build_voice_graphrag_prompt_parts`` is the preferred
Python spelling and is an alias with the same behavior.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
import string
from typing import Any, TypedDict


class VoiceGraphRagPromptParts(TypedDict):
    """Stable, JSON-safe input envelope for an injected GraphRAG backend."""

    system: str
    query: str
    language: str | None
    context: dict[str, Any]
    grounding: Any
    max_results: int


class VoiceTemplateError(ValueError):
    """Base error for invalid or unsafe voice-template input."""


class VoiceTemplateValidationError(VoiceTemplateError):
    """Raised when a template contains unsupported formatting expressions."""


class VoiceGroundingValidationError(VoiceTemplateError):
    """Raised when a grounded template cannot be safely rendered."""


_FIELD_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\((?:https?|ipfs)://[^)]+\)", re.I)
_CITATION_LINK_RE = re.compile(
    r"\[(?:source|sources|citation|citations)(?:\s+\d+)?\]\((?:https?|ipfs)://[^)]+\)",
    re.I,
)
_BRACKET_CITATION_RE = re.compile(r"\s*\[\s*\d+(?:\s*,\s*\d+)*\s*\]", re.I)
_URL_RE = re.compile(r"(?i)(?:https?|ipfs)://[^\s<>\]\)]+")
_CID_RE = re.compile(r"(?i)\bbafy[a-z0-9]{20,}\b")
_SOURCE_TAIL_RE = re.compile(
    r"(?is)\s+(?:sources?|evidence|citations?)\s*:\s*"
    r"(?:(?:https?|ipfs)://\S+|bafy[a-z0-9]+).*$"
)
_SSML_TAG_RE = re.compile(r"<\s*/?\s*[a-z][^>]*>", re.I)

_SYSTEM_PROMPT = (
    "You retrieve an Abby public-service response plan. Return a response "
    "template and structured slots only; never write an uncited final answer. "
    "Bind factual slots only to current evidence supplied by the caller, "
    "preserve source identifiers, and fail closed when evidence is missing "
    "or contradictory."
)


def _json_safe(value: Any) -> Any:
    """Copy a value through canonical JSON without importing schema modules."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_json_safe(item) for item in sorted(value, key=repr)]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    # Prompt payloads are diagnostic/context data.  A stable representation is
    # preferable to allowing an arbitrary object to execute during formatting.
    return repr(value)


def _canonical_copy(value: Any) -> Any:
    try:
        return json.loads(
            json.dumps(
                _json_safe(value),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as error:
        raise VoiceTemplateError(f"value is not JSON-safe: {error}") from error


def build_voice_graphrag_prompt_parts(
    transcript: str,
    *,
    context: Mapping[str, Any] | None = None,
    language: str | None = None,
    grounding: Mapping[str, Any] | Sequence[Any] | None = None,
    max_results: int = 5,
) -> VoiceGraphRagPromptParts:
    """Build a deterministic GraphRAG query envelope.

    The envelope is safe to pass to injected collaborators and is intentionally
    not an answer.  Context and grounding are copied, sorted through canonical
    JSON, so a backend cannot mutate the caller's request and cache identities
    do not depend on mapping insertion order.
    """

    query = str(transcript or "").strip()
    if not query:
        raise VoiceTemplateError("transcript must be non-empty")
    if isinstance(max_results, bool) or not isinstance(max_results, int) or max_results < 1:
        raise VoiceTemplateError("max_results must be a positive integer")
    if context is not None and not isinstance(context, Mapping):
        raise VoiceTemplateError("context must be a mapping")
    if grounding is not None and (
        isinstance(grounding, (str, bytes, bytearray))
        or not isinstance(grounding, (Mapping, Sequence))
    ):
        raise VoiceTemplateError("grounding must be a mapping or sequence")

    context_copy = _canonical_copy(dict(context or {}))
    grounding_copy = _canonical_copy(grounding or {})
    normalized_language = str(language).strip() if language is not None else ""
    return {
        "system": _SYSTEM_PROMPT,
        "query": query,
        "language": normalized_language or None,
        "context": context_copy,
        "grounding": grounding_copy,
        "max_results": max_results,
    }


def buildVoiceGraphRagPromptParts(
    transcript: str,
    *,
    context: Mapping[str, Any] | None = None,
    language: str | None = None,
    grounding: Mapping[str, Any] | Sequence[Any] | None = None,
    max_results: int = 5,
) -> VoiceGraphRagPromptParts:
    """Compatibility spelling for :func:`build_voice_graphrag_prompt_parts`."""

    return build_voice_graphrag_prompt_parts(
        transcript,
        context=context,
        language=language,
        grounding=grounding,
        max_results=max_results,
    )


def template_fields(template: str) -> tuple[str, ...]:
    """Return simple named placeholders accepted by the voice renderer."""

    text = str(template or "")
    if not text.strip():
        raise VoiceTemplateValidationError("template must be non-empty")
    fields: list[str] = []
    try:
        parsed = string.Formatter().parse(text)
        for _, field, format_spec, conversion in parsed:
            if field is None:
                continue
            if not _FIELD_RE.fullmatch(field):
                raise VoiceTemplateValidationError(
                    f"unsupported template placeholder {field!r}"
                )
            if format_spec or conversion:
                raise VoiceTemplateValidationError(
                    f"formatting is not allowed for {field!r}"
                )
            if field not in fields:
                fields.append(field)
    except VoiceTemplateValidationError:
        raise
    except (KeyError, IndexError, ValueError) as error:
        raise VoiceTemplateValidationError(f"invalid template: {error}") from error
    return tuple(fields)


def normalize_spoken_text(text: str) -> str:
    """Remove visual citations and markup before TTS synthesis.

    Source IDs and evidence remain in the structured router provenance.  A
    citation-only or empty response is rejected so the caller can use its
    deterministic safe fallback instead of synthesizing silence.
    """

    spoken = str(text or "")
    spoken = _CITATION_LINK_RE.sub("", spoken)
    spoken = _MARKDOWN_LINK_RE.sub(r"\1", spoken)
    spoken = _BRACKET_CITATION_RE.sub("", spoken)
    spoken = _SOURCE_TAIL_RE.sub("", spoken)
    spoken = _URL_RE.sub("", spoken)
    spoken = _CID_RE.sub("", spoken)
    spoken = _SSML_TAG_RE.sub("", spoken)
    spoken = re.sub(r"[ \t]+", " ", spoken)
    spoken = re.sub(r"\s+([,.;:!?])", r"\1", spoken)
    spoken = re.sub(r"([.!?])(?:\s*\1)+", r"\1", spoken)
    spoken = re.sub(r"\s*\n+\s*", " ", spoken).strip(" \t\r\n-")
    if not spoken:
        raise VoiceGroundingValidationError(
            "empty_spoken_response_after_citation_stripping"
        )
    return spoken


__all__ = [
    "VoiceGraphRagPromptParts",
    "VoiceTemplateError",
    "VoiceTemplateValidationError",
    "VoiceGroundingValidationError",
    "build_voice_graphrag_prompt_parts",
    "buildVoiceGraphRagPromptParts",
    "template_fields",
    "normalize_spoken_text",
]
