"""Malicious repository admission policy (EAAEF-121)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final


class RepositoryPolicyError(ValueError):
    """Repository is unsafe to onboard or execute."""


_FORBIDDEN_NAMES: Final[frozenset[str]] = frozenset(
    {
        ".git/hooks/pre-commit",
        "docker.sock",
        "id_rsa",
        ".env",
    }
)
_MAX_FILE_BYTES: Final[int] = 8 * 1024 * 1024
_MAX_FILES: Final[int] = 4096


def admit_repository(root: Path | str) -> None:
    """Read-only walk; refuse hooks, symlink escape, sockets, huge trees."""

    base = Path(root).resolve()
    if not base.is_dir():
        raise RepositoryPolicyError("repository root is not a directory")
    count = 0
    for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
        current = Path(dirpath)
        if current.is_symlink():
            raise RepositoryPolicyError("symlink directory is unsafe")
        for name in list(dirnames) + list(filenames):
            path = current / name
            relative = path.relative_to(base).as_posix()
            count += 1
            if count > _MAX_FILES:
                raise RepositoryPolicyError("repository exceeds file bound")
            if path.is_symlink():
                target = os.path.realpath(path)
                if not target.startswith(str(base)):
                    raise RepositoryPolicyError("symlink escape")
            if relative in _FORBIDDEN_NAMES or relative.endswith("/docker.sock"):
                raise RepositoryPolicyError(f"forbidden path {relative}")
            if "/hooks/" in f"/{relative}/" or relative.startswith(".git/hooks/"):
                raise RepositoryPolicyError("hooks must be disabled")
            if path.is_file() and not path.is_symlink() and path.stat().st_size > _MAX_FILE_BYTES:
                raise RepositoryPolicyError("file exceeds size bound")
            if path.is_socket() or path.name == "docker.sock":
                raise RepositoryPolicyError("unix socket is forbidden")
