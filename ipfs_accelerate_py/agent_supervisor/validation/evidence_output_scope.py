"""Fail-closed scope rules for objective evidence that is a concrete file.

Objective evidence is usually descriptive and therefore cannot grant edit
authority.  A small subset of evidence requirements are exact repository file
paths whose creation is itself part of the objective.  This module keeps that
exception typed, canonical, and shared by the objective planner and execution
daemon.
"""

from __future__ import annotations

import os
import posixpath
import re
from pathlib import PurePosixPath
from typing import Any, Iterable


EVIDENCE_OUTPUTS_METADATA_KEY = "evidence outputs"
_VCS_CONTROL_PATH_PARTS = frozenset(
    {
        ".git",
        ".gitmodules",
        ".hg",
        ".svn",
    }
)


def _canonical_repo_relative_path(
    value: Any,
    *,
    require_file_suffix: bool,
) -> str:
    """Return one exact POSIX repository path, or ``""`` when unsafe.

    The task board is a comma-delimited authority document, so glob patterns,
    URI-like values, platform-dependent paths, and non-canonical spellings are
    rejected instead of being repaired.  Requiring a suffix for evidence
    outputs prevents directory-shaped prose such as ``operator/approval`` from
    silently becoming file-write authority.
    """

    if not isinstance(value, (str, os.PathLike)):
        return ""
    raw = os.fspath(value)
    if not isinstance(raw, str):
        return ""
    path = raw.strip()
    if (
        not path
        or path != raw
        or path in {".", ".."}
        or path.startswith(("/", "\\", "~"))
        or path.endswith(("/", "\\"))
        or "\\" in path
        or "\0" in path
        or "://" in path
        or re.match(r"^[A-Za-z]:", path)
        or any(character in path for character in "*?[]")
        or any(ord(character) < 32 or ord(character) == 127 for character in path)
        or posixpath.normpath(path) != path
    ):
        return ""
    parts = PurePosixPath(path).parts
    if (
        not parts
        or any(part in {"", ".", ".."} for part in parts)
        or any(part.casefold() in _VCS_CONTROL_PATH_PARTS for part in parts)
    ):
        return ""
    if require_file_suffix and not PurePosixPath(path).suffix:
        return ""
    return path


def normalize_evidence_output_path(value: Any) -> str:
    """Return a safe exact evidence-output file path, or ``""``."""

    return _canonical_repo_relative_path(value, require_file_suffix=True)


def evidence_output_path_is_excluded(
    path: str,
    excluded_paths: Iterable[Any],
) -> bool:
    """Return whether ``path`` is an excluded control input or its descendant."""

    candidate = normalize_evidence_output_path(path)
    if not candidate:
        return True
    for value in excluded_paths:
        excluded = _canonical_repo_relative_path(
            value,
            require_file_suffix=False,
        )
        if excluded and (
            candidate == excluded or candidate.startswith(excluded + "/")
        ):
            return True
    return False


def split_evidence_output_values(value: Any) -> tuple[str, ...]:
    """Split one taskboard field without treating prose as path authority."""

    if not isinstance(value, str) or not value.strip():
        return ()
    values = tuple(item.strip() for item in value.split(","))
    if any(not item for item in values):
        return ()
    return values
