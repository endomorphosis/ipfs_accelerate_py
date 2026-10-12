"""Map one exact compact-view edit back to an original-source candidate.

This is byte-coordinate mapping, not semantic validation or filesystem access.
An unchanged view is authenticated by the existing codec before its KEEP map
is used. The candidate must project to the exact requested edited view under
the same codec profile. Transformed Python therefore passes the codec's syntax
checks, while identity views may contain arbitrary bytes. Neither case proves
behavior, grants mutation authority, or establishes live workspace freshness.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from .reversible_source_view import build_source_view, restore_source_view

SCHEMA = "supervisor-source-view-edit-candidate@1"
MAX_SOURCE_BYTES = 1024 * 1024
MAX_EDIT_BYTES = 64 * 1024
MAX_MANIFEST_BYTES = 512 * 1024


class SourceViewEditError(ValueError):
    """The edit cannot be mapped unambiguously under its pinned codec."""


def _need(value, message):
    if not value:
        raise SourceViewEditError(message)


def _wire(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise SourceViewEditError("canonical source-view edit metadata required") from exc


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class SourceViewEdit:
    """Inert result; manifest and receipt access returns independent copies."""

    candidate: bytes
    view: bytes
    manifest_json: bytes
    original_start: int
    original_end: int
    receipt_json: bytes

    @property
    def manifest(self):
        return json.loads(self.manifest_json)

    @property
    def manifest_sha256(self):
        return _sha(self.manifest_json)

    @property
    def receipt(self):
        return json.loads(self.receipt_json)


def _original_range(original, view, pieces, start, end):
    if not original and not view and start == end == 0:
        return 0, 0
    offset = 0
    for piece in pieces:
        if piece["kind"] != "keep":
            continue
        size = piece["end"] - piece["start"]
        if start < end and offset <= start < end <= offset + size:
            return piece["start"] + start - offset, piece["start"] + end - offset
        if start == end:
            # At a collapsed DELETE boundary there are two possible original
            # anchors. Only strict interiors and exposed outer endpoints map.
            inside = offset < start < offset + size
            beginning = start == offset == 0 and piece["start"] == 0
            ending = start == offset + size == len(view) and piece["end"] == len(original)
            if inside or beginning or ending:
                anchor = piece["start"] + start - offset
                return anchor, anchor
        offset += size
    raise SourceViewEditError("edit crosses a hidden region or has an ambiguous insertion boundary")


def map_source_view_edit(*, original: bytes, view: bytes, manifest: dict,
        expected_manifest_sha256: str, view_start: int, view_end: int,
        expected_view_preimage: bytes, replacement: bytes) -> SourceViewEdit:
    """Return one exact candidate without writing or executing target source.

    Nonempty ranges must fit wholly within one retained KEEP interval. Inserts
    at collapsed hidden text refuse, including hidden BOF/EOF regions. Existing
    caller-protected line numbers require both preimage and replacement to omit
    LF entirely. Other edits may change line count if exact codec replay succeeds.
    """
    _need(type(original) is bytes and type(view) is bytes
        and len(original) <= MAX_SOURCE_BYTES and len(view) <= MAX_SOURCE_BYTES,
        "bounded exact original and view bytes required")
    _need(type(expected_view_preimage) is bytes and type(replacement) is bytes
        and len(expected_view_preimage) <= MAX_EDIT_BYTES and len(replacement) <= MAX_EDIT_BYTES,
        "bounded exact edit preimage and replacement bytes required")
    _need(type(view_start) is int and type(view_end) is int
        and 0 <= view_start <= view_end <= len(view), "bounded exact view byte offsets required")
    _need(type(manifest) is dict, "exact source-view manifest required")
    manifest_raw = _wire(manifest)
    _need(len(manifest_raw) <= MAX_MANIFEST_BYTES, "source-view manifest byte bound exceeded")
    retained = json.loads(manifest_raw)
    # Authenticate the complete original/view/map before relying on any map
    # coordinate, even when a caller recomputed a forged manifest's hash.
    try:
        restored = restore_source_view(original=original, view=view, manifest=retained,
            expected_manifest_sha256=expected_manifest_sha256)
    except (ValueError, TypeError, KeyError, UnicodeError, RecursionError) as exc:
        raise SourceViewEditError("unchanged source-view authentication failed") from exc
    _need(restored == original, "unchanged source-view inverse differs")
    _need(view[view_start:view_end] == expected_view_preimage, "exact edit preimage differs")
    _need(expected_view_preimage != replacement, "no-op source-view edit refused")
    if retained["protected_lines"]:
        _need(b"\n" not in expected_view_preimage and b"\n" not in replacement,
            "edit changes caller-protected physical line numbers")
    transformed_python = retained["effective_profile"] in {"python-whitespace@1", "python-comments@1"}
    if transformed_python:
        _need(all(not (0 < offset < len(view) and view[offset - 1:offset + 1] == b"\r\n")
            for offset in (view_start, view_end)), "transformed Python edit splits a CRLF boundary")
        try:
            view[:view_start].decode("utf-8", "strict")
            view[:view_end].decode("utf-8", "strict")
            replacement.decode("utf-8", "strict")
        except UnicodeError as exc:
            raise SourceViewEditError("transformed Python edits require UTF-8 character boundaries") from exc
    original_start, original_end = _original_range(original, view, retained["pieces"], view_start, view_end)
    candidate_bytes = len(original) - (original_end - original_start) + len(replacement)
    _need(candidate_bytes <= MAX_SOURCE_BYTES, "candidate source byte bound exceeded")
    candidate = original[:original_start] + replacement + original[original_end:]
    expected_view = view[:view_start] + replacement + view[view_end:]
    rebuilt = build_source_view(candidate, profile=retained["requested_profile"],
        protected_lines=tuple(retained["protected_lines"]))
    _need(all(rebuilt.manifest[key] == retained[key] for key in
        ("requested_profile", "effective_profile", "fallback_reason", "codec_binding", "protected_lines")),
        "edit changes the source-view profile, fallback or codec binding")
    _need(rebuilt.view == expected_view, "edited view does not exactly match candidate codec projection")
    rebuilt_manifest = _wire(rebuilt.manifest)
    _need(len(rebuilt_manifest) <= MAX_MANIFEST_BYTES, "rebuilt source-view manifest byte bound exceeded")
    receipt = dict(schema=SCHEMA, candidate_only=True,
        original_sha256=_sha(original), original_bytes=len(original),
        view_sha256=_sha(view), view_bytes=len(view), manifest_sha256=_sha(manifest_raw),
        view_start=view_start, view_end=view_end, original_start=original_start, original_end=original_end,
        preimage_sha256=_sha(expected_view_preimage), preimage_bytes=len(expected_view_preimage),
        replacement_sha256=_sha(replacement), replacement_bytes=len(replacement),
        candidate_sha256=_sha(candidate), candidate_bytes=len(candidate),
        candidate_view_sha256=_sha(rebuilt.view), candidate_view_bytes=len(rebuilt.view),
        candidate_manifest_sha256=_sha(rebuilt_manifest),
        requested_profile=retained["requested_profile"], effective_profile=retained["effective_profile"],
        fallback_reason=retained["fallback_reason"],
        exact_compact_edit_round_trip=True, untouched_original_bytes_preserved=True,
        python_syntax_checked=transformed_python, source_freshness_verified=False,
        source_semantics_verified=False, semantic_equivalence_claimed=False,
        proof_authority=False, completion_authority=False, execution_authority=False,
        mutation_authority=False, omission_authority=False, filesystem_written=False,
        source_executed=False, source_obligations_discharged=0, provider_calls=0,
        model_inference_calls=0, solver_calls=0, prover_calls=0,
        interpretation="Exact byte mapping and codec replay only; identity views have no syntax guarantee. "
            "Caller owns workspace freshness, patch admission and behavioral validation.")
    return SourceViewEdit(candidate, rebuilt.view, rebuilt_manifest, original_start, original_end, _wire(receipt))


__all__ = ["SourceViewEdit", "SourceViewEditError", "map_source_view_edit", "SCHEMA",
    "MAX_SOURCE_BYTES", "MAX_EDIT_BYTES"]
