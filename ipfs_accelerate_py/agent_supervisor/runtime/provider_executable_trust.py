"""Dependency-free host provider executable trust boundaries.

These helpers are used while deciding whether a provider route is ready.
Keeping them independent of the provider runners prevents readiness checks
from importing execution-only native dependency closures.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def resolve_codex_quota_fallback_executable(
    *,
    workspace: str | Path,
    configured: str = "",
) -> str:
    """Resolve a pinned executable that the Grok workspace cannot replace."""

    workspace_path = Path(workspace).expanduser().resolve()
    codex_candidate = str(configured or shutil.which("codex") or "").strip()
    if not codex_candidate:
        return ""
    candidate_path = Path(codex_candidate).expanduser()
    if not candidate_path.is_absolute():
        resolved_from_path = shutil.which(codex_candidate)
        if not resolved_from_path:
            return ""
        candidate_path = Path(resolved_from_path)
    try:
        resolved_candidate = candidate_path.resolve(strict=True)
    except OSError:
        return ""
    candidate_entry = Path(os.path.abspath(candidate_path))
    system_entries = {
        Path("/usr/bin/codex"),
        Path("/usr/local/bin/codex"),
        Path("/usr/bin/codex.exe"),
        Path("/usr/local/bin/codex.exe"),
    }
    package_roots = (
        Path("/usr/lib/node_modules/@openai/codex"),
        Path("/usr/local/lib/node_modules/@openai/codex"),
    )
    matched_root = next(
        (
            root
            for root in package_roots
            if resolved_candidate == root
            or resolved_candidate.is_relative_to(root)
        ),
        resolved_candidate.parent
        if resolved_candidate.parent in {Path("/usr/bin"), Path("/usr/local/bin")}
        else None,
    )
    try:
        trust_chain = (
            [candidate_entry, candidate_entry.parent, resolved_candidate]
            + (
                list(resolved_candidate.parents)[
                    : list(resolved_candidate.parents).index(matched_root) + 1
                ]
                if matched_root is not None and resolved_candidate != matched_root
                else ([matched_root] if matched_root is not None else [])
            )
        )
        trusted_chain = all(
            path.lstat().st_uid == 0
            and (path.is_symlink() or not path.stat().st_mode & 0o022)
            for path in trust_chain
        )
    except (OSError, ValueError):
        trusted_chain = False
    if (
        candidate_entry not in system_entries
        or matched_root is None
        or not trusted_chain
        or not candidate_entry.is_file()
        or not os.access(candidate_entry, os.X_OK)
        or candidate_entry.is_relative_to(workspace_path)
        or resolved_candidate.is_relative_to(workspace_path)
        or candidate_entry.name.casefold() not in {"codex", "codex.exe"}
    ):
        return ""
    return str(candidate_entry)


__all__ = ["resolve_codex_quota_fallback_executable"]
