#!/usr/bin/env python3
"""Remove unused `import asyncio` statements.

This repo is mid-migration from asyncio to AnyIO; a common mechanical failure
mode is leaving behind `import asyncio` after all `asyncio.*` usages were
converted.

This script conservatively removes *only* bare `import asyncio` lines when the
module no longer references `asyncio.` anywhere.

Usage:
  python tools/remove_unused_asyncio_imports.py --check
  python tools/remove_unused_asyncio_imports.py --apply

Exit codes:
  0: no changes needed / successful apply
  1: --check found files that would change
  2: unexpected failure
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List


SKIP_DIRS = {".venv", ".venv_zt_validate", ".pytest_cache", "__pycache__", ".git"}


def iter_py_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.py"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if not path.is_file():
            continue
        yield path


def process_file(path: Path) -> bool:
    """Return True if file would change."""
    try:
        original = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return False

    if "import asyncio" not in original:
        return False

    # If asyncio is still referenced, don't touch imports.
    if "asyncio." in original:
        return False

    lines = original.splitlines(True)
    changed = False
    new_lines: List[str] = []

    for line in lines:
        if line.strip() == "import asyncio":
            changed = True
            continue
        new_lines.append(line)

    if not changed:
        return False

    updated = "".join(new_lines)
    if updated == original:
        return False

    path.write_text(updated, encoding="utf-8")
    return True


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args(argv)

    root = Path(args.root)

    changed_paths: List[Path] = []
    for path in iter_py_files(root):
        if args.check:
            # simulate by editing in-memory
            try:
                original = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if "import asyncio" not in original:
                continue
            if "asyncio." in original:
                continue
            if any(line.strip() == "import asyncio" for line in original.splitlines()):
                changed_paths.append(path)
        else:
            if process_file(path):
                changed_paths.append(path)

    if args.check:
        if changed_paths:
            print(f"Would remove unused import asyncio in {len(changed_paths)} files")
            for p in changed_paths[:50]:
                print(f"  - {p.relative_to(root)}")
            if len(changed_paths) > 50:
                print(f"  ... ({len(changed_paths) - 50} more)")
            return 1
        print("No unused asyncio imports found")
        return 0

    print(f"Removed unused asyncio imports in {len(changed_paths)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
