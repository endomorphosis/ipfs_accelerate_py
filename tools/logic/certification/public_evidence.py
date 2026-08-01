"""Portable, fail-closed projections for checked-in certification evidence.

Certification receipts are often assembled from live tool output.  That
output may contain host-specific executable, repository, or temporary paths
which must not participate in the digest of a public artifact.  This module is
deliberately dependency-free so focused certifiers can apply the same policy
before computing their outer receipt digest.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence, Set
from pathlib import Path
from typing import Any, Final

MAX_PUBLIC_STRING_BYTES: Final = 8192
HOST_PATH_REDACTION: Final = "<host-path-redacted>"
REPO_ROOT_REDACTION: Final = "<repo-root>"

# Match path-shaped fragments rooted in common private/user-specific
# locations.  The broad final component intentionally covers paths embedded
# in prover diagnostics; quote/whitespace boundaries keep surrounding prose.
HOST_PRIVATE_PATH_RE: Final = re.compile(
    r"(?<![A-Za-z0-9._-])"
    r"/(?:private/tmp|tmp|home|Users)"
    r"(?:/[^\s\"'<>]*)?"
)

RAW_OUTPUT_KEYS: Final = frozenset(
    {"stdout", "stderr", "raw_stdout", "raw_stderr"}
)
RAW_SECRET_KEYS: Final = frozenset(
    {
        "secret",
        "private_secret",
        "private_witness",
        "witness",
        "witness_bytes",
        "trapdoor",
        "toxic_waste",
        "private_key",
        "api_key",
        "access_token",
    }
)

_TRAILING_PATH_PUNCTUATION: Final = ".,;:!?)]}"


def _repo_root_texts(repo_root: str | Path | None) -> tuple[str, ...]:
    if repo_root is None:
        return ()
    root = Path(repo_root).expanduser()
    candidates = {str(root.absolute()), str(root.resolve())}
    return tuple(
        sorted(
            (
                candidate.rstrip("/\\")
                for candidate in candidates
                if candidate not in {"", "/", "\\"}
            ),
            key=len,
            reverse=True,
        )
    )


def _path_basename(path: str) -> str:
    candidate = path.rstrip("/\\")
    if not candidate:
        return ""
    return candidate.rsplit("/", 1)[-1]


def _useful_path_basename(path: str) -> str:
    parts = [part for part in path.strip("/\\").split("/") if part]
    if not parts:
        return ""
    # A home directory's first child is normally a user name, not useful
    # public evidence.  Preserve only deeper artifact/executable basenames.
    if parts[0] in {"home", "Users"} and len(parts) <= 2:
        return ""
    return _path_basename(path)


def _redact_host_match(match: re.Match[str]) -> str:
    path = match.group(0)
    trailing = ""
    while path and path[-1] in _TRAILING_PATH_PUNCTUATION:
        trailing = path[-1] + trailing
        path = path[:-1]
    basename = _useful_path_basename(path)
    marker = HOST_PATH_REDACTION
    if basename and basename not in {"home", "Users", "tmp"}:
        marker = f"{marker}/{basename}"
    return marker + trailing


def _redact_paths(value: str, *, repo_root: str | Path | None) -> str:
    projected = value
    for root_text in _repo_root_texts(repo_root):
        projected = re.sub(
            re.escape(root_text) + r"(?=$|[/\\])",
            REPO_ROOT_REDACTION,
            projected,
        )
    return HOST_PRIVATE_PATH_RE.sub(_redact_host_match, projected)


def _contains_raw_secret_marker(value: str) -> bool:
    lowered = value.lower()
    return any(
        marker in lowered
        for marker in (
            "private-witness-fvt047-secret-axiom-never-leak",
            "witness_bytes=",
            "toxic_waste=",
            "trapdoor=",
            "private_secret=",
            "secret=",
        )
    )


def _path_only_projection(
    value: Any,
    *,
    repo_root: str | Path | None,
) -> Any:
    """Project paths before hashing a value hidden behind a redaction row."""

    if isinstance(value, Mapping):
        return {
            _redact_paths(str(key), repo_root=repo_root): _path_only_projection(
                child,
                repo_root=repo_root,
            )
            for key, child in value.items()
        }
    if isinstance(value, (Set, Sequence)) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        items = value
        if isinstance(value, Set):
            items = sorted(value, key=repr)
        return [
            _path_only_projection(child, repo_root=repo_root)
            for child in items
        ]
    if isinstance(value, Path):
        return _redact_paths(str(value), repo_root=repo_root)
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, str):
        return _redact_paths(value, repo_root=repo_root)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _redact_paths(str(value), repo_root=repo_root)


def _redacted_value(
    value: Any,
    *,
    reason: str,
    repo_root: str | Path | None,
) -> dict[str, Any]:
    portable = _path_only_projection(value, repo_root=repo_root)
    encoded = json.dumps(
        portable,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return {
        "redacted": True,
        "reason": reason,
        "byte_length": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _project_public_evidence(
    value: Any,
    *,
    repo_root: str | Path | None,
    key_name: str,
) -> Any:
    normalized_key = key_name.lower()
    if normalized_key in RAW_OUTPUT_KEYS:
        if isinstance(value, Mapping) and value.get("redacted") is True:
            return {
                _redact_paths(
                    str(key),
                    repo_root=repo_root,
                ): _project_public_evidence(
                    child,
                    repo_root=repo_root,
                    key_name=str(key),
                )
                for key, child in value.items()
            }
        return _redacted_value(
            value,
            reason="raw_process_output_forbidden",
            repo_root=repo_root,
        )
    if normalized_key in RAW_SECRET_KEYS and value not in (None, True, False):
        if isinstance(value, Mapping) and value.get("redacted") is True:
            return {
                _redact_paths(
                    str(key),
                    repo_root=repo_root,
                ): _project_public_evidence(
                    child,
                    repo_root=repo_root,
                    key_name=str(key),
                )
                for key, child in value.items()
            }
        return _redacted_value(
            value,
            reason="raw_secret_or_witness_forbidden",
            repo_root=repo_root,
        )
    if isinstance(value, Mapping):
        return {
            _redact_paths(str(key), repo_root=repo_root): _project_public_evidence(
                child,
                repo_root=repo_root,
                key_name=str(key),
            )
            for key, child in value.items()
        }
    if isinstance(value, (Set, Sequence)) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        items = value
        if isinstance(value, Set):
            items = sorted(value, key=repr)
        return [
            _project_public_evidence(
                child,
                repo_root=repo_root,
                key_name="",
            )
            for child in items
        ]
    if isinstance(value, bytes):
        return _redacted_value(
            value.hex(),
            reason="raw_bytes_forbidden",
            repo_root=repo_root,
        )
    if isinstance(value, Path):
        value = str(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if not isinstance(value, str):
        value = str(value)
    if _contains_raw_secret_marker(value):
        return _redacted_value(
            value,
            reason="raw_secret_or_witness_marker_forbidden",
            repo_root=repo_root,
        )

    projected = _redact_paths(value, repo_root=repo_root)
    if len(projected.encode("utf-8")) > MAX_PUBLIC_STRING_BYTES:
        return _redacted_value(
            projected,
            reason="unbounded_public_string_forbidden",
            repo_root=repo_root,
        )
    return projected


def public_evidence_projection(
    value: Any,
    *,
    repo_root: str | Path | None = None,
) -> Any:
    """Return a recursively portable, bounded, JSON-compatible value."""

    return _project_public_evidence(
        value,
        repo_root=repo_root,
        key_name="",
    )


def public_evidence_audit(
    value: Any,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Audit an already-projected value and fail closed on unsafe evidence."""

    failures: list[str] = []
    root_texts = _repo_root_texts(repo_root)

    def audit_string(text: str) -> None:
        if HOST_PRIVATE_PATH_RE.search(text):
            failures.append("host_private_path")
        if any(root_text in text for root_text in root_texts):
            failures.append("repo_root_path")
        if _contains_raw_secret_marker(text):
            failures.append("raw_secret_or_witness_marker")
        if len(text.encode("utf-8")) > MAX_PUBLIC_STRING_BYTES:
            failures.append("unbounded_public_string")

    def walk(item: Any, key: str = "") -> None:
        normalized_key = key.lower()
        if normalized_key in RAW_OUTPUT_KEYS:
            if not (
                isinstance(item, Mapping)
                and item.get("redacted") is True
                and item.get("sha256")
            ):
                failures.append(f"raw_process_output:{key}")
            elif isinstance(item, Mapping):
                for child_key, child in item.items():
                    audit_string(str(child_key))
                    walk(child, str(child_key))
            return
        if normalized_key in RAW_SECRET_KEYS and item not in (None, True, False):
            if not (
                isinstance(item, Mapping)
                and item.get("redacted") is True
                and item.get("sha256")
            ):
                failures.append(f"raw_secret_or_witness:{key}")
            elif isinstance(item, Mapping):
                for child_key, child in item.items():
                    audit_string(str(child_key))
                    walk(child, str(child_key))
            return
        if isinstance(item, Mapping):
            for child_key, child in item.items():
                audit_string(str(child_key))
                walk(child, str(child_key))
            return
        if isinstance(item, (Set, Sequence)) and not isinstance(
            item, (str, bytes, bytearray)
        ):
            for child in item:
                walk(child)
            return
        if isinstance(item, Path):
            audit_string(str(item))
            return
        if isinstance(item, bytes):
            failures.append("raw_bytes")
            return
        if isinstance(item, str):
            audit_string(item)
            return
        if item is not None and not isinstance(item, (bool, int, float)):
            failures.append("non_json_value")

    walk(value)
    violations = sorted(set(failures))
    safe = not violations
    return {
        "satisfied": safe,
        "safe": safe,
        "failures": violations,
        "violations": violations,
        "violation_count": len(violations),
        "host_private_paths_forbidden": True,
        "repo_root_paths_forbidden": True,
        "raw_process_output_forbidden": True,
        "raw_secret_or_witness_forbidden": True,
        "max_public_string_bytes": MAX_PUBLIC_STRING_BYTES,
    }


__all__ = [
    "HOST_PATH_REDACTION",
    "HOST_PRIVATE_PATH_RE",
    "MAX_PUBLIC_STRING_BYTES",
    "RAW_OUTPUT_KEYS",
    "RAW_SECRET_KEYS",
    "REPO_ROOT_REDACTION",
    "public_evidence_audit",
    "public_evidence_projection",
]
