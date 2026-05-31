"""Reusable wrapper helpers for accelerator-backed supervisor entry points."""

from __future__ import annotations

import os
import shlex
import shutil
import sys
from pathlib import Path
from typing import Iterable, Sequence


def with_default(argv: Sequence[str], flag: str, value: str) -> list[str]:
    """Prepend a default flag/value pair unless the caller already provided it."""

    args = list(argv)
    if flag in args:
        return args
    return [flag, value, *args]


def with_flag_default(argv: Sequence[str], flag: str) -> list[str]:
    """Prepend a default boolean flag unless the caller already provided it."""

    args = list(argv)
    if flag in args:
        return args
    return [flag, *args]


def with_repeated_default(argv: Sequence[str], flag: str, values: Iterable[str]) -> list[str]:
    """Prepend repeated default flag/value pairs unless the caller already provided the flag."""

    args = list(argv)
    if flag in args:
        return args
    defaults: list[str] = []
    for value in values:
        defaults.extend([flag, str(value)])
    return [*defaults, *args]


def default_llm_merge_resolver_command(
    *,
    primary_env_var: str = "",
    fallback_env_var: str = "IPFS_ACCELERATE_AGENT_LLM_MERGE_RESOLVER_COMMAND",
    codex_args: Sequence[str] = ("exec", "--dangerously-bypass-approvals-and-sandbox", "-C", ".", "-"),
) -> str:
    """Return the configured merge-resolver command, falling back to Codex when available."""

    if primary_env_var:
        configured = os.environ.get(primary_env_var, "").strip()
        if configured:
            return configured
    if fallback_env_var:
        configured = os.environ.get(fallback_env_var, "").strip()
        if configured:
            return configured
    codex = shutil.which("codex")
    if not codex:
        return ""
    return " ".join(shlex.quote(part) for part in (codex, *codex_args))


def _unique_path_entries(entries: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for entry in entries:
        if not entry or entry in seen:
            continue
        seen.add(entry)
        unique.append(entry)
    return unique


def ensure_runtime_pythonpath(paths: Sequence[Path | str], *, env_var: str = "PYTHONPATH") -> None:
    """Make local package roots importable for the current process and child processes."""

    normalized_paths = [str(Path(path)) for path in paths]
    for path in reversed(normalized_paths):
        if path not in sys.path:
            sys.path.insert(0, path)

    existing = os.environ.get(env_var, "")
    existing_paths = existing.split(os.pathsep) if existing else []
    os.environ[env_var] = os.pathsep.join(_unique_path_entries([*normalized_paths, *existing_paths]))
