"""Cross-process helpers for durable taskboard allocation and updates."""

from __future__ import annotations

import fcntl
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TextIO


@contextmanager
def locked_taskboard(path: Path) -> Iterator[TextIO]:
    """Lock a taskboard's inode while a scanner performs read-modify-write."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            stream.seek(0)
            yield stream
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def replace_locked_taskboard(stream: TextIO, text: str) -> None:
    """Replace and durably flush a taskboard held by ``locked_taskboard``."""

    stream.seek(0)
    stream.truncate()
    stream.write(text)
    stream.flush()
    os.fsync(stream.fileno())


def task_ids_from_artifact_names(
    directory: Path,
    *,
    task_prefix: str,
) -> set[str]:
    """Recover allocated display IDs from durable discovery filenames."""

    if not directory.exists():
        return set()
    normalized = task_prefix.rstrip("-") + "-"
    pattern = re.compile(
        rf"(?<![A-Za-z0-9]){re.escape(normalized)}(?P<number>\d+)(?!\d)",
        flags=re.IGNORECASE,
    )
    task_ids: set[str] = set()
    for path in directory.rglob("*"):
        if not path.is_file():
            continue
        for match in pattern.finditer(path.name):
            number = int(match.group("number"))
            task_ids.add(f"{normalized}{number:03d}")
    return task_ids
