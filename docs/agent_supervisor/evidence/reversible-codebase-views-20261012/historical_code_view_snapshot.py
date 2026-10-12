"""Conservative, read-only Python display views; never candidate rewriting.

AST/token checks establish the stated structural checks only. They do not
authenticate source, prove semantic equivalence, or authorize task completion.
"""
from __future__ import annotations

import ast
import codecs
import hashlib
import io
import json
import re
import token
import tokenize
from typing import Callable

SCHEMA = "supervisor-code-view-preview@1"
MAX_SOURCE_BYTES = 2_000_000
_STRING_TOKEN_TYPES = {
    token.STRING,
    *(getattr(token, name) for name in ("FSTRING_START", "FSTRING_MIDDLE", "FSTRING_END") if hasattr(token, name)),
}
GUIDANCE = (
    "READ-ONLY CODE VIEW: ordinary comments/trailing whitespace may be omitted; "
    "physical source line numbers are unchanged. Read exact source before edits. "
    "AST/token structure checks are not a semantic-equivalence proof or task authority.\n"
)


class CodeViewMinificationError(ValueError):
    """The source or preview could not satisfy the closed display contract."""


# Inspect COMMENT tokens only, never text inside Python STRING tokens.
_PROTECTED_COMMENT = re.compile(
    r"(?ix)"
    r"(?:coding\s*[:=]|copyright|\blicen[cs]e\b|spdx|all\s+rights\s+reserved|"
    r"permission\s+is\s+hereby|redistribution|\bwarranty\b|"
    r"\b(?:noqa|nosec|pragma|pylint|pyright|mypy|ruff|bandit|coverage|"
    r"cython|pyre|pytype|suppress|fmt|isort|yapf|autopep8|doctest|"
    r"region|endregion|TODO|FIXME|XXX|HACK|BUG)\b|\btype\s*:|"
    r"\b(?:security|vulnerab\w*|insecure|attack\w*|secret\w*|password\w*|"
    r"auth\w*|escape\w*|sanitize\w*|invariant\w*|contract\w*|"
    r"required|must|shall|never)\b|\bdo\s+not\b)"
)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _tokens(source: str) -> list[tokenize.TokenInfo]:
    try:
        result = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError) as exc:
        raise CodeViewMinificationError("source tokenization failed") from exc
    if any(item.type == token.ERRORTOKEN for item in result):
        raise CodeViewMinificationError("source contains an unsupported Python token")
    return result


def _structure(source: str) -> str:
    try:
        # type_ignores/type_comments are retained in the comparison too.
        tree = ast.parse(source, type_comments=True)
        return ast.dump(tree, annotate_fields=True, include_attributes=False)
    except (SyntaxError, ValueError, TypeError, RecursionError) as exc:
        raise CodeViewMinificationError("source AST structure cannot be checked") from exc


def _meaning(tokens, removed=()):
    removed = set(removed)
    return [
        (item.type, item.string)
        for index, item in enumerate(tokens)
        if item.type != token.NL and index not in removed
    ]


def _local_count(counter, source):
    try:
        value = counter(source)
    except Exception as exc:
        raise CodeViewMinificationError("local token counter failed") from exc
    if type(value) is not int or value < 0:
        raise CodeViewMinificationError("local token counter must return a nonnegative integer")
    return value


def build_code_view_preview(
    source_text: str, *, token_counter: Callable[[str], int] | None = None,
    max_bytes: int = 1_000_000,
) -> dict:
    """Remove only ordinary COMMENT tokens and safe trailing display trivia.

    Physical lines and meaningful token spellings are preserved. Leading header
    comments, shebangs, encoding/license notices, and conservative tooling,
    typing/security/contract directives remain exact. No source is executed or
    written. The caller must independently bind source and count its full final
    model input; this preview is not a cumulative provider-usage estimate.
    """
    if type(max_bytes) is not int or not 1 <= max_bytes <= MAX_SOURCE_BYTES:
        raise CodeViewMinificationError("max_bytes must be a bounded positive integer")
    if not isinstance(source_text, str):
        raise CodeViewMinificationError("source_text must be a string")
    if token_counter is not None and not callable(token_counter):
        raise CodeViewMinificationError("token_counter must be callable")
    try:
        raw = source_text.encode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise CodeViewMinificationError("source is not strict UTF-8") from exc
    if len(raw) > max_bytes:
        raise CodeViewMinificationError("source exceeds max_bytes")
    try:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(raw).readline)
        if codecs.lookup(encoding).name not in {"utf-8", "utf-8-sig"}:
            raise CodeViewMinificationError("source declares a non-UTF-8 encoding")
    except (SyntaxError, LookupError) as exc:
        raise CodeViewMinificationError("source encoding declaration is invalid") from exc

    original_structure = _structure(source_text)
    original_tokens = _tokens(source_text)
    # StringIO splits only physical LF lines, not Unicode characters that
    # str.splitlines also treats as separators inside Python string literals.
    lines = list(io.StringIO(source_text).readlines())
    starts, byte_starts = [], []
    char_offset = byte_offset = 0
    for line in lines:
        starts.append(char_offset)
        byte_starts.append(byte_offset)
        char_offset += len(line)
        byte_offset += len(line.encode("utf-8"))

    significant = {token.COMMENT, token.NL, token.NEWLINE, token.ENCODING, token.ENDMARKER}
    first_code_line = next((item.start[0] for item in original_tokens if item.type not in significant), len(lines) + 1)
    protected_comments = {
        index for index, item in enumerate(original_tokens)
        if item.type == token.COMMENT and (
            item.start[0] < first_code_line or item.string.startswith("#!")
            or _PROTECTED_COMMENT.search(item.string)
        )
    }
    # Preserve complete comment paragraphs containing a protected notice, not
    # merely the keyword-bearing line of a license or security directive.
    comment_blocks = []
    block = []
    previous_line = 0
    for index, item in enumerate(original_tokens):
        if item.type != token.COMMENT or lines[item.start[0] - 1][:item.start[1]].strip():
            continue
        line = item.start[0]
        if block and any(text.strip() for text in lines[previous_line:line - 1]):
            comment_blocks.append(block)
            block = []
        block.append(index)
        previous_line = line
    if block:
        comment_blocks.append(block)
    for block in comment_blocks:
        if any(index in protected_comments for index in block):
            protected_comments.update(block)
    protected_lines = set()
    removed_indices = []
    removals = []
    records = []
    for index, item in enumerate(original_tokens):
        if item.type in _STRING_TOKEN_TYPES:
            protected_lines.update(range(item.start[0], item.end[0] + 1))
        if item.type != token.COMMENT:
            continue
        if index in protected_comments:
            protected_lines.add(item.start[0])
            continue
        start_line, start_col = item.start
        end_line, end_col = item.end
        start = starts[start_line - 1] + start_col
        end = starts[end_line - 1] + end_col
        comment_raw = item.string.encode("utf-8")
        records.append({
            "start_line": start_line, "start_column": start_col,
            "end_line": end_line, "end_column": end_col,
            "start_utf8_byte": byte_starts[start_line - 1] + len(lines[start_line - 1][:start_col].encode("utf-8")),
            "end_utf8_byte": byte_starts[end_line - 1] + len(lines[end_line - 1][:end_col].encode("utf-8")),
            "sha256": _sha(comment_raw), "bytes": len(comment_raw),
        })
        removed_indices.append(index)
        removals.append((start, end))

    parts = []
    previous_end = 0
    for start, end in removals:
        parts.append(source_text[previous_end:start])
        previous_end = end
    parts.append(source_text[previous_end:])
    candidate = "".join(parts)
    candidate_lines = list(io.StringIO(candidate).readlines())
    trimmed_bytes = 0
    for index, line in enumerate(candidate_lines, 1):
        if index in protected_lines:
            continue
        newline = "\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""
        body = line[:-len(newline)] if newline else line
        trimmed = body.rstrip(" \t\f")
        trimmed_bytes += len(body.encode("utf-8")) - len(trimmed.encode("utf-8"))
        candidate_lines[index - 1] = trimmed + newline
    candidate = "".join(candidate_lines)
    if len(candidate_lines) != len(lines):
        raise CodeViewMinificationError("display transformation changed physical line count")
    if _meaning(original_tokens, removed_indices) != _meaning(_tokens(candidate)):
        raise CodeViewMinificationError("display transformation changed meaningful tokens")
    if original_structure != _structure(candidate):
        raise CodeViewMinificationError("display transformation changed AST structure")

    candidate_rendered = GUIDANCE + candidate
    candidate_bytes = len(candidate_rendered.encode("utf-8"))
    selected = candidate != source_text and candidate_bytes < len(raw) and candidate_bytes <= max_bytes
    view = candidate if selected else source_text
    rendered = candidate_rendered if selected else source_text
    actual_records = records if selected else []
    records_raw = json.dumps(actual_records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    local_counts = None
    if token_counter is not None:
        local_counts = {
            "scope": "LOCAL caller tokenizer; complete strings independently; not cumulative provider usage",
            "original": _local_count(token_counter, source_text),
            "rendered_view": _local_count(token_counter, rendered),
        }
    return {
        "schema": SCHEMA, "selected": "minified" if selected else "original",
        "original_source_sha256": _sha(raw), "original_source_bytes": len(raw),
        "view_text": view, "view_sha256": _sha(view.encode("utf-8")), "view_bytes": len(view.encode("utf-8")),
        "guidance": GUIDANCE, "rendered_view": rendered, "rendered_bytes": len(rendered.encode("utf-8")),
        "candidate_rendered_bytes": candidate_bytes,
        "fallback_reason": None if selected else "complete_guided_view_does_not_reduce_bytes",
        "line_mapping": {"mode": "identity", "line_count": len(lines), "original_line_start": 1, "view_line_start": 1},
        "removed_comments": actual_records, "removed_comments_sha256": _sha(records_raw),
        "removed_comment_count": len(actual_records), "removed_comment_bytes": sum(item["bytes"] for item in actual_records),
        "removed_trailing_whitespace_bytes": trimmed_bytes if selected else 0,
        "ast_structure_verified": True, "token_structure_verified": True,
        "source_authentication_claimed": False, "semantic_equivalence_claimed": False,
        "proof_authority": False, "execution_authority": False, "completion_authority": False,
        "local_token_counts": local_counts,
    }
