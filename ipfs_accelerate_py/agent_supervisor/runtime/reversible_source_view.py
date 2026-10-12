"""Exact pinned source-display codec; no source execution or mutation.

Comment/token protection derives from the archived read-only helper at
docs/agent_supervisor/evidence/reversible-codebase-views-20261012/
historical_code_view_snapshot.py (SHA256 0c3ed9348189ed0bf5f45a1c11c24fc3a6acff9c45d9400c853181fe58e3bebd).
Restoration also needs original bytes; a deletion map cannot recover lost text.
Structural checks grant no semantic equivalence, omission or proof authority.
"""
from __future__ import annotations

import ast
import codecs
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import token
import tokenize

SCHEMA = "supervisor-reversible-source-view@1"
PROFILES = frozenset({"identity@1", "python-whitespace@1", "python-comments@1"})
MAX_TRANSFORM_BYTES = 1024 * 1024
MAX_PIECES = 4096
MAX_TOKENS = 131072
HISTORICAL_HELPER_SHA256 = "0c3ed9348189ed0bf5f45a1c11c24fc3a6acff9c45d9400c853181fe58e3bebd"
_FLAGS = ("source_authentication_claimed", "source_freshness_verified",
    "semantic_equivalence_claimed", "proof_authority", "completion_authority",
    "execution_authority", "mutation_authority", "omission_authority")
_FIELDS = {"schema", "requested_profile", "effective_profile", "fallback_reason",
    "original_sha256", "original_bytes", "view_sha256", "view_bytes",
    "protected_lines", "codec_binding", "pieces", "candidate_only", *_FLAGS}
_STRING_TYPES = {token.STRING, *(getattr(token, name) for name in
    ("FSTRING_START", "FSTRING_MIDDLE", "FSTRING_END") if hasattr(token, name))}
_PROTECTED = re.compile(
    r"(?ix)(?:coding\s*[:=]|copyright|\blicen[cs]e\b|spdx|all\s+rights\s+reserved|"
    r"permission\s+is\s+hereby|redistribution|\bwarranty\b|"
    r"\b(?:noqa|nosec|pragma|pylint|pyright|mypy|ruff|bandit|coverage|cython|pyre|"
    r"pytype|suppress|fmt|isort|yapf|autopep8|doctest|region|endregion|TODO|FIXME|"
    r"XXX|HACK|BUG)\b|\btype\s*:|\b(?:security|vulnerab\w*|insecure|attack\w*|"
    r"secret\w*|password\w*|auth\w*|escape\w*|sanitize\w*|invariant\w*|"
    r"contract\w*|required|must|shall|never)\b|\bdo\s+not\b)"
)


class SourceViewError(ValueError):
    """A codec argument, retained representation or exact inverse refused."""


class _Fallback(Exception):
    pass


def _need(condition, reason):
    if not condition:
        raise SourceViewError(reason)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, RecursionError, UnicodeError) as exc:
        raise SourceViewError("invalid canonical source-view manifest") from exc


def current_codec_binding() -> dict:
    """Local implementation byte/version identities, without authentication."""
    return {"schema": "supervisor-source-view-codec-binding@1", "codec_version": 1,
        "codec_sha256": _sha(Path(__file__).resolve(strict=True).read_bytes()),
        "historical_helper_sha256": HISTORICAL_HELPER_SHA256,
        "python_version": sys.version, "python_implementation": sys.implementation.name,
        "python_cache_tag": sys.implementation.cache_tag,
        **{name + "_sha256": _sha(Path(module.__file__).resolve(strict=True).read_bytes())
           for name, module in (("ast", ast), ("token", token), ("tokenize", tokenize))}}


@dataclass(frozen=True)
class SourceView:
    view: bytes
    manifest: dict

    @property
    def manifest_sha256(self):
        return _sha(_canonical(self.manifest))


def _arguments(source, profile, protected_lines):
    _need(type(source) is bytes, "source must be exact bytes")
    _need(type(profile) is str and profile in PROFILES, "unsupported source-view profile")
    _need(type(protected_lines) is tuple and len(protected_lines) <= MAX_PIECES
        and all(type(line) is int and line > 0 for line in protected_lines),
        "protected lines must be bounded positive physical line indices")
    _need(protected_lines == tuple(sorted(set(protected_lines))), "protected lines must be sorted and unique")
    count = source.count(b"\n") + int(bool(source) and not source.endswith(b"\n"))
    _need(not protected_lines or protected_lines[-1] <= count, "protected line is outside source")


def _tokens(text):
    result = []
    try:
        for item in tokenize.generate_tokens(io.StringIO(text).readline):
            if len(result) >= MAX_TOKENS or item.type == token.ERRORTOKEN:
                raise _Fallback("unsupported_tokens")
            result.append(item)
    except (tokenize.TokenError, IndentationError, SyntaxError) as exc:
        raise _Fallback("unsupported_tokens") from exc
    return result


def _structure(text):
    try:
        tree = ast.parse(text, type_comments=True)
        if any(isinstance(node, ast.JoinedStr) for node in ast.walk(tree)):
            raise _Fallback("unsupported_fstrings")
        return ast.dump(tree, include_attributes=False)
    except (SyntaxError, ValueError, TypeError, RecursionError) as exc:
        raise _Fallback("invalid_python") from exc


def _partition(size, removals):
    pieces = []
    cursor = 0
    for start, end in removals:
        if start > cursor:
            pieces.append({"kind": "keep", "start": cursor, "end": start})
        if pieces and pieces[-1]["kind"] == "delete" and pieces[-1]["end"] == start:
            pieces[-1]["end"] = end
        else:
            pieces.append({"kind": "delete", "start": start, "end": end})
        cursor = end
    if cursor < size:
        pieces.append({"kind": "keep", "start": cursor, "end": size})
    if len(pieces) > MAX_PIECES:
        raise _Fallback("piece_limit")
    return pieces


def _python_view(source, profile, protected_lines):
    if b"\r" in source.replace(b"\r\n", b""):
        raise _Fallback("unsupported_newlines")
    try:
        text = source.decode("utf-8", "strict")
    except UnicodeError as exc:
        raise _Fallback("non_utf8") from exc
    try:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(source).readline)
        if codecs.lookup(encoding).name not in {"utf-8", "utf-8-sig"}:
            raise _Fallback("unsupported_encoding")
    except (SyntaxError, LookupError) as exc:
        raise _Fallback("unsupported_encoding") from exc
    tokens = _tokens(text)
    structure = _structure(text)
    lines = list(io.StringIO(text).readlines())  # Physical LF, not Unicode splitlines.
    significant = {token.COMMENT, token.NL, token.NEWLINE, token.ENCODING, token.ENDMARKER}
    first_code = next((t.start[0] for t in tokens if t.type not in significant), len(lines) + 1)
    protected = {i for i, t in enumerate(tokens) if t.type == token.COMMENT and
        (profile == "python-whitespace@1" or t.start[0] < first_code or
         t.start[0] in protected_lines or t.string.startswith("#!") or _PROTECTED.search(t.string))}
    block, previous = [], 0
    for i, t in enumerate(tokens):
        if t.type != token.COMMENT or lines[t.start[0] - 1][:t.start[1]].strip():
            continue
        if block and any(line.strip() for line in lines[previous:t.start[0] - 1]):
            if any(index in protected for index in block):
                protected.update(block)
            block = []
        block.append(i)
        previous = t.start[0]
    if any(index in protected for index in block):
        protected.update(block)
    untouched = set(protected_lines)
    comments, removed = {}, set()
    for i, t in enumerate(tokens):
        if t.type in _STRING_TYPES:
            untouched.update(range(t.start[0], t.end[0] + 1))
        if t.type == token.COMMENT:
            if i in protected:
                untouched.add(t.start[0])
            else:
                comments[t.start[0]] = t
                removed.add(i)
    removals, offset = [], 0
    for number, line in enumerate(lines, 1):
        ending = "\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""
        body = line[:-len(ending)] if ending else line
        comment = comments.get(number)
        start = end = len(body)
        if comment:
            if comment.start[0] != comment.end[0] or comment.end[1] != len(body):
                raise _Fallback("unsupported_tokens")
            start, end = comment.start[1], comment.end[1]
        if number not in untouched:
            start = len(body[:start].rstrip(" \t\f"))
        if start < end:
            removals.append((offset + len(body[:start].encode()), offset + len(body[:end].encode())))
        offset += len(line.encode())
    pieces = _partition(len(source), removals)
    view = b"".join(source[p["start"]:p["end"]] for p in pieces if p["kind"] == "keep")
    candidate = view.decode("utf-8")
    meaning = lambda ts, omitted: [(t.type, t.string) for i, t in enumerate(ts)
        if t.type != token.NL and i not in omitted]
    if (len(list(io.StringIO(candidate).readlines())) != len(lines)
            or meaning(tokens, removed) != meaning(_tokens(candidate), set())
            or _structure(candidate) != structure):
        raise _Fallback("unsupported_tokens")
    return view, pieces


def build_source_view(source: bytes, *, profile="python-whitespace@1", protected_lines=()) -> SourceView:
    """Build a candidate display and complete original-byte KEEP/DELETE map."""
    _arguments(source, profile, protected_lines)
    binding = current_codec_binding()
    view, pieces, effective, reason = source, _partition(len(source), []), profile, None
    if profile != "identity@1":
        try:
            if len(source) > MAX_TRANSFORM_BYTES:
                raise _Fallback("source_too_large")
            view, pieces = _python_view(source, profile, protected_lines)
        except _Fallback as exc:
            view, pieces, effective, reason = source, _partition(len(source), []), "identity@1", str(exc)
    manifest = dict(schema=SCHEMA, requested_profile=profile, effective_profile=effective,
        fallback_reason=reason, original_sha256=_sha(source), original_bytes=len(source),
        view_sha256=_sha(view), view_bytes=len(view), protected_lines=list(protected_lines),
        codec_binding=binding, pieces=pieces, candidate_only=True, **{flag: False for flag in _FLAGS})
    _need(current_codec_binding() == binding, "codec implementation changed during projection")
    return SourceView(view, manifest)


def restore_source_view(*, original: bytes, view: bytes, manifest: dict, expected_manifest_sha256: str) -> bytes:
    """Restore the unchanged pinned view; arbitrary edited views refuse."""
    _need(type(original) is bytes and type(view) is bytes and type(manifest) is dict,
          "exact source/view bytes and manifest required")
    _need(type(expected_manifest_sha256) is str and re.fullmatch("[0-9a-f]{64}", expected_manifest_sha256),
          "independently expected manifest SHA256 required")
    raw = _canonical(manifest)
    retained = json.loads(raw)
    _need(_sha(raw) == expected_manifest_sha256 and set(retained) == _FIELDS
        and retained["schema"] == SCHEMA and retained["candidate_only"] is True
        and all(retained[flag] is False for flag in _FLAGS), "retained source-view manifest differs")
    _need(type(retained["protected_lines"]) is list, "invalid retained protected lines")
    _arguments(original, retained["requested_profile"], tuple(retained["protected_lines"]))
    _need(retained["original_sha256"] == _sha(original) and retained["original_bytes"] == len(original)
        and retained["view_sha256"] == _sha(view) and retained["view_bytes"] == len(view), "source/view pins differ")
    pieces = retained["pieces"]
    _need(type(pieces) is list and len(pieces) <= MAX_PIECES, "invalid bounded source-view pieces")
    cursor, previous = 0, None
    for piece in pieces:
        _need(type(piece) is dict and set(piece) == {"kind", "start", "end"}
            and type(piece["kind"]) is str and piece["kind"] in {"keep", "delete"}
            and type(piece["start"]) is int and type(piece["end"]) is int
            and piece["start"] == cursor and cursor < piece["end"] <= len(original)
            and piece["kind"] != previous, "source-view partition is incomplete or noncanonical")
        cursor, previous = piece["end"], piece["kind"]
    _need(cursor == len(original), "source-view partition omits bytes")
    _need(retained["codec_binding"] == current_codec_binding(), "current codec binding differs")
    rebuilt = build_source_view(original, profile=retained["requested_profile"],
                                protected_lines=tuple(retained["protected_lines"]))
    _need(rebuilt.view == view and _canonical(rebuilt.manifest) == raw, "source-view profile replay differs")
    parts, view_cursor = [], 0
    for piece in pieces:
        start, end = piece["start"], piece["end"]
        if piece["kind"] == "keep":
            segment = view[view_cursor:view_cursor + end - start]
            _need(segment == original[start:end], "retained view bytes differ")
            parts.append(segment)
            view_cursor += end - start
        else:
            parts.append(original[start:end])
    restored = b"".join(parts)
    _need(view_cursor == len(view) and restored == original, "source-view inverse differs")
    _need(current_codec_binding() == retained["codec_binding"], "codec changed during restoration")
    return restored


__all__ = ["SourceView", "SourceViewError", "build_source_view", "restore_source_view", "current_codec_binding"]
