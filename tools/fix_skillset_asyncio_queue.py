#!/usr/bin/env python3
"""Fix leftover asyncio.Queue references in runtime skillset modules.

This repo is migrating from asyncio to AnyIO/AnyioQueue.
Some skillset modules still contain placeholder return statements like:

    return endpoint, processor, endpoint_handler, # TODO: ... - asyncio.Queue(32), 0

Because the TODO is a comment, the function actually returns only 3 values.
This script converts those placeholders into real returns using AnyioQueue,
updates docstrings mentioning asyncio.Queue, and ensures AnyioQueue is imported.

Scope: only ipfs_accelerate_py/worker/skillset/*.py (runtime, not tests).
"""

from __future__ import annotations

import re
from pathlib import Path


SKILLSET_DIR = Path(__file__).resolve().parents[1] / "ipfs_accelerate_py" / "worker" / "skillset"

RETURN_TODO_RE = re.compile(
    r"^(?P<indent>[ \t]*)return(?P<prefix>.*?)\s*,\s*#\s*TODO:\s*Replace\s*with\s*anyio\.create_memory_object_stream\s*-\s*asyncio\.Queue\((?P<size>\d+)\)\s*,\s*(?P<tail>.*)$"
)

ASSIGN_TODO_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<lhs>[A-Za-z_][A-Za-z0-9_]*\s*=)\s*#\s*TODO:\s*Replace\s*with\s*anyio\.create_memory_object_stream\s*-\s*asyncio\.Queue\((?P<size>\d+)\)\s*$"
)


def ensure_anyioqueue_import(text: str) -> str:
    if "AnyioQueue" not in text:
        return text

    if re.search(r"^\s*from\s+\.\.anyio_queue\s+import\s+AnyioQueue\b", text, re.MULTILINE):
        return text

    lines = text.splitlines(keepends=True)

    # Prefer inserting after the first `import anyio` to keep imports tidy.
    for idx, line in enumerate(lines):
        if re.match(r"^\s*import\s+anyio\b", line):
            lines.insert(idx + 1, "from ..anyio_queue import AnyioQueue\n")
            return "".join(lines)

    # Fallback: insert after the last top-level import.
    last_import_idx = -1
    for idx, line in enumerate(lines):
        if line.startswith("import ") or line.startswith("from "):
            last_import_idx = idx
            continue
        # Stop once we leave the import block.
        if last_import_idx != -1 and line.strip() and not line.startswith("#"):
            break

    insert_at = last_import_idx + 1 if last_import_idx != -1 else 0
    lines.insert(insert_at, "from ..anyio_queue import AnyioQueue\n")
    return "".join(lines)


def fix_file(path: Path) -> bool:
    raw = path.read_text(encoding="utf-8")
    updated = raw

    # Update docstrings/comments that reference asyncio.Queue.
    updated = updated.replace("asyncio.Queue", "AnyioQueue")

    # Convert placeholder returns into real returns.
    def _return_sub(m: re.Match[str]) -> str:
        indent = m.group("indent")
        prefix = m.group("prefix")
        size = m.group("size")
        tail = m.group("tail")
        # Preserve original spacing in the return prefix.
        return f"{indent}return{prefix}, AnyioQueue({size}), {tail}"

    updated = RETURN_TODO_RE.sub(_return_sub, updated)

    # Convert placeholder assignments.
    updated = ASSIGN_TODO_RE.sub(
        lambda m: f"{m.group('indent')}{m.group('lhs')} AnyioQueue({m.group('size')})\n",
        updated,
    )

    updated = ensure_anyioqueue_import(updated)

    if updated != raw:
        path.write_text(updated, encoding="utf-8")
        return True

    return False


def main() -> int:
    if not SKILLSET_DIR.exists():
        raise SystemExit(f"Skillset directory not found: {SKILLSET_DIR}")

    changed = 0
    for py_file in sorted(SKILLSET_DIR.glob("*.py")):
        if fix_file(py_file):
            changed += 1

    print(f"Updated {changed} file(s) under {SKILLSET_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
